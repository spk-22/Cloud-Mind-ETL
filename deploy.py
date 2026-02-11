from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential

# -------------------------------
# Step 0: Connect to workspace
# -------------------------------
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="Put your ID here",
    resource_group_name="AI-ETL",
    workspace_name="cloud_ML_training"
)

# -------------------------------
# Step 1: Create online endpoint
# -------------------------------
endpoint_name = "dynamic-forecast"

# Check if endpoint exists
try:
    ml_client.online_endpoints.get(endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists.")
except Exception:
    print(f"Creating endpoint '{endpoint_name}'...")
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode="key"  # key-based authentication
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint '{endpoint_name}' created.")

# -------------------------------
# Step 2: Get your registered model
# -------------------------------
model = ml_client.models.get(
    name="demandForecast",
    version=1  # use version 1 since that is the one you registered
)

# -------------------------------
# Step 3: Create deployment
# -------------------------------
deployment_name = "blue"  # you can name it anything
deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=endpoint_name,
    model=model,
    environment="new-dyncprice:1",  # your curated/custom environment
    instance_type="Standard_DS3_v2",  # minimum recommended SKU
    instance_count=1,
    liveness_probe_timeout_seconds=120,
    readiness_probe_timeout_seconds=120
)

print(f"Deploying model '{model.name}' to endpoint '{endpoint_name}'...")
ml_client.online_deployments.begin_create_or_update(deployment).result()
print(f"Deployment '{deployment_name}' completed.")

# -------------------------------
# Step 4: Route traffic to deployment
# -------------------------------
endpoint = ml_client.online_endpoints.get(endpoint_name)
endpoint.traffic = {deployment_name: 100}  # route 100% traffic to this deployment
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"Traffic routed: 100% to deployment '{deployment_name}'")

# -------------------------------
# Step 5: Ready to test
# -------------------------------
print(f"Endpoint URL: {endpoint.scoring_uri}")
print("Use the keys from Azure portal to authenticate and call the /score API")
