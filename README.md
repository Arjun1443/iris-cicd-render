# Iris Classification API with CI and Optional Render Deployment

A Flask inference API accompanied by model training, tests, and a GitHub Actions workflow.

## Why this project matters

Model development becomes more useful when inference is exposed through a documented interface and changes are checked automatically. This project connects training, request validation, API testing, and an optional deployment trigger.

## Local setup

```bash
git clone https://github.com/Arjun1443/iris-cicd-render.git
cd iris-cicd-render
python -m venv .venv
# Activate .venv before continuing.
python -m pip install -r requirements.txt
python train_model.py
python -m pytest test_app.py -v
python app.py
```

Run commands from the repository root because the app loads `iris_model.pkl` by a relative path.

## API example

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"features":[5.1,3.5,1.4,0.2]}'
```

Feature order: sepal length, sepal width, petal length, petal width. The API returns a `prediction` label. GET requests accept `f1` through `f4`; malformed or missing feature inputs return validation errors.

## Automation

`.github/workflows/cicd.yml` installs dependencies, trains the model, and runs tests. On a push to `main`, the deployment job runs only when repository variable `ENABLE_RENDER_DEPLOY` is `true` and Render credentials are configured.

The existence of deployment configuration does not establish that a public endpoint is live.

## Technical focus

Python · Flask · NumPy · scikit-learn · joblib · pytest · GitHub Actions · deployment integration

## Limitations

This is an Iris demonstration model. Dependencies are not version-pinned, serialized model compatibility depends on the training environment, and the development server is not a production serving configuration.
