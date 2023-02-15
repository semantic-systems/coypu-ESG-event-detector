# coypu-EventExtraction
Event Extraction module - Master of Disaster (MoD) - for deployment.

This repo contains both the event type detection and event argument extraction modules for inference. Currently, for the first two MVPs, these two modules are independently implemented, meaning it is error pruned - event type and arguments could be mismatched. 
The plan is to merge this two modules into one in the last MVP.

- [x] MVP 1: custom Event Detector Event Argument Extractor v1
- [x] MVP 2: custom Event Detector Event Argument Extractor v2
- [ ] MVP 3: HiTeC Event Extractor

# Requirement
- python >= 3.7

Use your favorite virtual environment management tool and 
```
pip install -r requirements.txt
```

# Usage

## Local Gradio Deployment
Users can create a simple gradio demo where MoD is hosted locally. After the module is instantiated, two links will be generated.
One local link and another a public link. 
```
python event_extractor.py 
```

## Docker
Alternatively, you can deploy via docker, which will be using port 5278. 
```
docker-compose up
```

and query via post request,
```
curl -i -H “Content-Type: application/json” -X POST -d ‘{“message”: “there was an earthquake in Hamburg last night man damn hot noodles.“}’ http://127.0.0.1:5278
```
