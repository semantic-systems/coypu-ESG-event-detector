from event_extractor import EventExtractor
from flask import abort, Flask, jsonify, request
from flask_healthz import healthz
from models.event_argument_extraction.OpenTapiocaArgumentExtractor import OpenTapiocaArgumentExtractor
from models.event_detection.EventDetector import EventDetector
from models.event_detection.ESGEventDetector import ESGEventDetector

app = Flask(__name__)

app.register_blueprint(healthz, url_prefix="/healthz")


def liveness():
    pass


def readiness():
    pass


app.config.update(
    HEALTHZ = {
        "live": app.name + ".liveness",
        "ready": app.name + ".readiness"
    }
)
event_detector = ESGEventDetector()
event_extractor = EventExtractor(event_detector=event_detector, event_argument_extractor=None)


@app.route('/', methods=['POST'])
def flask():
    if not request.json or not 'message' in request.json:
        print(request.json)
        abort(400)

    message = request.json['message']

    output = event_extractor.infer(message)

    response = {'message': message, 'event type': output[0], 'wikidata link': output[1]}
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5279)

