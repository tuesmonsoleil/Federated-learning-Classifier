from flask import Flask, request, jsonify, send_file
import pickle
import gzip
from io import BytesIO
import torch
import logging

app = Flask(__name__)

model_parameters = []
avg_params = None
epoch = -1

# Setup logging
logging.basicConfig(level=logging.DEBUG)

def average_model_parameters(param_list):
    if not param_list:
        return None
    avg_params = {}
    for key in param_list[0].keys():
        avg_params[key] = torch.mean(torch.stack([p[key] for p in param_list]), dim=0)
    return avg_params

@app.route('/upload', methods=['POST'])
def upload_parameters():
    global avg_params, model_parameters, epoch

    if 'model' not in request.files:
        logging.error("No model file in request")
        return "No model file in request", 400

    model_file = request.files['model']
    try:
        with gzip.GzipFile(fileobj=model_file.stream, mode='rb') as f:
            parameters = pickle.load(f)
    except (OSError, pickle.UnpicklingError) as e:
        logging.error(f"Error handling the file: {str(e)}")
        return f"Error handling the file: {str(e)}", 400

    model_parameters.append(parameters)
    logging.info(f"Received parameters from client. Total received: {len(model_parameters)}")

    if len(model_parameters) >= 2:
        avg_params = average_model_parameters(model_parameters)
        epoch += 1
        model_parameters.clear()
        logging.info(f"Parameters averaged for epoch {epoch}")
        return jsonify({"status": "Parameters averaged"}), 200

    return jsonify({"status": "Parameter received"}), 200

@app.route('/model', methods=['GET'])
def get_model():
    global avg_params, epoch
    client_epoch = int(request.args.get('epoch'))

    if avg_params is None or epoch != client_epoch:
        logging.error(f"No averaged model parameters available for epoch {client_epoch}")
        return "No averaged model parameters available.", 404

    memfile = BytesIO()
    with gzip.GzipFile(fileobj=memfile, mode='wb') as f:
        pickle.dump(avg_params, f)
    memfile.seek(0)

    logging.info(f"Sending averaged parameters for epoch {epoch}")
    return send_file(memfile, download_name='model.pkl.gz', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
