# Hosted MLflow UI on Render

## Purpose

This Docker deployment rebuilds MLflow runs and a registered model from the public UCI Bike Sharing data **during image build**. After build, MLflow serves the real Tracking UI and Model Registry over HTTPS through Render. This is different from a static report or a localhost URL.

## One-time setup (Render)

1. Sign in at https://dashboard.render.com/ with GitHub.
2. Select **New > Blueprint** and connect `bettyjam729-prog/capital-bikeshare-mlflow`. Render reads `render.yaml`.
3. Choose the web service. Select a name, confirm **Free** if available. Supply `MLFLOW_AUTH_ADMIN_PASSWORD` as a random password of at least 12 characters. Render generates the Flask secret key.
4. Create/deploy the service; the Docker build installs MLflow, runs tests, downloads `day.csv`, trains/evaluates models, and registers the selected model.
5. Open the generated `https://<service-name>.onrender.com` address and log in with username **admin** and the password you entered.
6. Check **Experiments** (three candidate training runs and one final run), **Models** (registered model), and selected run's data and metrics. Share the URL with your instructor and provide credentials **privately**, if required.

## Important

- This is a public HTTPS **address**, but the UI requires login to protect the MLflow REST API. Do **not** put the admin password in GitHub, README or the submitted ZIP.
- Render Free sleeps after inactivity and may require ~1 minute to wake. Its filesystem is ephemeral. Experiments are regenerated when the service is rebuilt, and further edits to the live experiment DB are not durable.
- For persistent team tracking, migrate to a paid persistent disk or a managed external database plus artifact storage.
- This is a short-lived **coursework demonstration**, not an internet-facing production MLflow deployment.
- Rebuilding reruns training and creates new Run IDs, so do not reuse earlier local run IDs when describing the hosted instance.
- To submit your repository ZIP, include this DEPLOYMENT.md and add the **actual** Render URL to README only after the Render deployment succeeds.
