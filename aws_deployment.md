# AWS Deployment CLI Guide — Flan-T5 Summariser

> **All-CLI, no Console.** Every command below runs in a terminal.
> Complete the variables block once, then copy-paste each section in order.

---

## 0. Variables — Fill in once, export, reference everywhere

```bash
# ── Identity ────────────────────────────────────────────────────────────────
export AWS_ACCOUNT_ID="123456789012"          # from: aws sts get-caller-identity
export AWS_REGION="us-east-1"

# ── ECR ─────────────────────────────────────────────────────────────────────
export ECR_REPO="flan-t5-summarizer"
export ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ── App Runner ───────────────────────────────────────────────────────────────
export SERVICE_NAME="flan-t5-summarizer"
export SERVICE_ARN=""                         # fill after Section 5

# ── Model & observability ────────────────────────────────────────────────────
export HF_MODEL_ID="google/flan-t5-base"
export HF_TOKEN="hf_..."                      # Hugging Face access token
export WANDB_API_KEY="..."                    # Weights & Biases API key
export WANDB_PROJECT="flan-t5-summarizer"     # matches logger.py default
```

---

## 1. Prerequisites

### AWS CLI v2

```bash
# macOS
brew install awscli

# Linux (x86_64)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install

# Verify
aws --version
# Expected: aws-cli/2.x.x Python/3.x ...
```

### Docker with buildx

```bash
# Docker Desktop already includes buildx.
# Verify:
docker buildx version
# Expected: github.com/docker/buildx v0.x.x ...
```

### Configure AWS credentials

```bash
aws configure
# AWS Access Key ID [None]: <your-key-id>
# AWS Secret Access Key [None]: <your-secret>
# Default region name [None]: us-east-1
# Default output format [None]: json
```

### Verify identity

```bash
aws sts get-caller-identity
# Returns: Account, UserId, Arn
# Copy Account value → set AWS_ACCOUNT_ID above
```

---

## 2. IAM — Create a deployment user

> Skip this section if you are already authenticated as an admin locally.
> The user created here is intended for CI/CD (GitHub Actions) only.

### Create user

```bash
aws iam create-user --user-name flan-t5-deployer
```

### Create and attach an inline policy

```bash
# Write the policy document
cat > /tmp/flan-t5-deploy-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRAuth",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ECRRepository",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:CompleteLayerUpload",
        "ecr:GetDownloadUrlForLayer",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart",
        "ecr:BatchGetImage",
        "ecr:DescribeImages",
        "ecr:ListImages"
      ],
      "Resource": "arn:aws:ecr:${AWS_REGION}:${AWS_ACCOUNT_ID}:repository/flan-t5-summarizer"
    },
    {
      "Sid": "AppRunner",
      "Effect": "Allow",
      "Action": [
        "apprunner:CreateService",
        "apprunner:DescribeService",
        "apprunner:ListServices",
        "apprunner:StartDeployment",
        "apprunner:DeleteService",
        "apprunner:UpdateService",
        "apprunner:TagResource"
      ],
      "Resource": "*"
    }
  ]
}
EOF

# Expand shell variables into the policy file
envsubst < /tmp/flan-t5-deploy-policy.json > /tmp/flan-t5-deploy-policy-resolved.json

# Attach as an inline policy
aws iam put-user-policy \
  --user-name flan-t5-deployer \
  --policy-name flan-t5-deploy-policy \
  --policy-document file:///tmp/flan-t5-deploy-policy-resolved.json
```

### Create access keys — save the output for GitHub Secrets

```bash
aws iam create-access-key --user-name flan-t5-deployer
# Output contains:
#   AccessKeyId     → AWS_ACCESS_KEY_ID secret
#   SecretAccessKey → AWS_SECRET_ACCESS_KEY secret  (shown ONCE — save it now)
```

---

## 3. ECR — Create the repository

```bash
aws ecr create-repository \
  --repository-name $ECR_REPO \
  --region $AWS_REGION \
  --image-scanning-configuration scanOnPush=true \
  --image-tag-mutability MUTABLE
```

Expected output (excerpt):

```json
{
  "repository": {
    "repositoryUri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/flan-t5-summarizer",
    ...
  }
}
```

Confirm `ECR_REGISTRY` matches the `repositoryUri` prefix:

```bash
echo $ECR_REGISTRY
# 123456789012.dkr.ecr.us-east-1.amazonaws.com
```

---

## 4. Docker — Authenticate, build, push

### Authenticate Docker to ECR

```bash
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ECR_REGISTRY
# Expected: Login Succeeded
```

### Build and push (linux/amd64 — matches App Runner runtime)

```bash
docker buildx build \
  --platform linux/amd64 \
  -t $ECR_REGISTRY/$ECR_REPO:latest \
  -f inference/Dockerfile \
  . \
  --push
```

> The Dockerfile copies `inference/` and `pyproject.toml`/`uv.lock` from the
> repo root, so the build context must be the project root (`.`).

### Verify the image is in ECR

```bash
aws ecr list-images \
  --repository-name $ECR_REPO \
  --region $AWS_REGION
```

---

## 5. App Runner — Create the service

### Write the service configuration

```bash
cat > /tmp/apprunner-service.json << EOF
{
  "ServiceName": "${SERVICE_NAME}",
  "SourceConfiguration": {
    "ImageRepository": {
      "ImageIdentifier": "${ECR_REGISTRY}/${ECR_REPO}:latest",
      "ImageConfiguration": {
        "Port": "8080",
        "RuntimeEnvironmentVariables": {
          "HF_MODEL_ID":    "${HF_MODEL_ID}",
          "HF_TOKEN":       "${HF_TOKEN}",
          "WANDB_API_KEY":  "${WANDB_API_KEY}",
          "WANDB_PROJECT":  "${WANDB_PROJECT}"
        }
      },
      "ImageRepositoryType": "ECR"
    },
    "AutoDeploymentsEnabled": false
  },
  "InstanceConfiguration": {
    "Cpu": "1 vCPU",
    "Memory": "2 GB"
  },
  "HealthCheckConfiguration": {
    "Protocol": "HTTP",
    "Path": "/health",
    "Interval": 20,
    "Timeout": 5,
    "HealthyThreshold": 1,
    "UnhealthyThreshold": 5
  }
}
EOF
```

### Create the service

```bash
aws apprunner create-service \
  --cli-input-json file:///tmp/apprunner-service.json \
  --region $AWS_REGION
```

Expected output (excerpt):

```json
{
  "Service": {
    "ServiceArn": "arn:aws:apprunner:us-east-1:123456789012:service/flan-t5-summarizer/abcdef1234",
    "ServiceUrl": "abcdef1234.us-east-1.awsapprunner.com",
    "Status": "OPERATION_IN_PROGRESS",
    ...
  }
}
```

**Save the `ServiceArn` now:**

```bash
export SERVICE_ARN="arn:aws:apprunner:us-east-1:123456789012:service/flan-t5-summarizer/abcdef1234"
```

### Poll until the service is RUNNING

```bash
watch -n 15 "aws apprunner describe-service \
  --service-arn $SERVICE_ARN \
  --region $AWS_REGION \
  --query 'Service.Status' \
  --output text"
# Wait for: RUNNING
```

### Verify the health endpoint

```bash
SERVICE_URL=$(aws apprunner describe-service \
  --service-arn $SERVICE_ARN \
  --region $AWS_REGION \
  --query 'Service.ServiceUrl' \
  --output text)

curl -sf "https://${SERVICE_URL}/health"
# Expected: {"status":"ok"}  (or similar)
```

---

## 6. Manual redeploy — push new image and trigger

Use this when you need to deploy outside of CI/CD.

### Re-authenticate (tokens expire after 12 hours)

```bash
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ECR_REGISTRY
```

### Build, tag with SHA, and push

```bash
IMAGE_TAG=$(git rev-parse --short HEAD)

docker buildx build \
  --platform linux/amd64 \
  -t $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG \
  -t $ECR_REGISTRY/$ECR_REPO:latest \
  -f inference/Dockerfile \
  . \
  --push

echo "Pushed: $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG"
```

### Trigger App Runner deployment

```bash
aws apprunner start-deployment \
  --service-arn $SERVICE_ARN \
  --region $AWS_REGION
```

### Poll deployment status

```bash
aws apprunner describe-service \
  --service-arn $SERVICE_ARN \
  --region $AWS_REGION \
  --query 'Service.{Status:Status,UpdatedAt:UpdatedAt}' \
  --output table
# Wait for Status: RUNNING
```

---

## 7. GitHub Actions secrets setup

These 7 secrets must exist in the repository before the
`.github/workflows/deploy.yml` workflow will succeed.

```bash
# Replace each "..." with the actual value.

gh secret set AWS_ACCESS_KEY_ID      --body "AKIA..."
gh secret set AWS_SECRET_ACCESS_KEY  --body "..."
gh secret set AWS_REGION             --body "us-east-1"
gh secret set ECR_REPOSITORY         --body "flan-t5-summarizer"
gh secret set APP_RUNNER_SERVICE_ARN --body "arn:aws:apprunner:us-east-1:123456789012:service/flan-t5-summarizer/abcdef1234"
gh secret set HF_TOKEN               --body "hf_..."
gh secret set WANDB_API_KEY          --body "..."
```

Verify all secrets are present:

```bash
gh secret list
```

---

## 8. Monitoring and debugging

### Service status

```bash
aws apprunner describe-service \
  --service-arn $SERVICE_ARN \
  --region $AWS_REGION \
  --query 'Service.{Status:Status,Url:ServiceUrl,CPU:InstanceConfiguration.Cpu,Memory:InstanceConfiguration.Memory}' \
  --output table
```

### List all App Runner services

```bash
aws apprunner list-services --region $AWS_REGION
```

### CloudWatch logs

App Runner streams logs to CloudWatch automatically.

```bash
# Find the log group
aws logs describe-log-groups \
  --log-group-name-prefix /aws/apprunner/$SERVICE_NAME \
  --region $AWS_REGION \
  --query 'logGroups[*].logGroupName' \
  --output table
```

```bash
# Tail application logs in real time (replace <instance-id> with the log stream)
aws logs tail \
  /aws/apprunner/$SERVICE_NAME/<instance-id>/application \
  --region $AWS_REGION \
  --follow
```

```bash
# Or tail the service-level (deployment/health) logs
aws logs tail \
  /aws/apprunner/$SERVICE_NAME/<instance-id>/service \
  --region $AWS_REGION \
  --follow
```

> Tip: run `aws logs describe-log-streams --log-group-name /aws/apprunner/$SERVICE_NAME/<instance-id>/application --region $AWS_REGION` to list available stream names.

### Health check

```bash
SERVICE_URL=$(aws apprunner describe-service \
  --service-arn $SERVICE_ARN \
  --region $AWS_REGION \
  --query 'Service.ServiceUrl' \
  --output text)

curl -v "https://${SERVICE_URL}/health"
```

---

## 9. Rollback to a previous image

### List available ECR image tags

```bash
aws ecr describe-images \
  --repository-name $ECR_REPO \
  --region $AWS_REGION \
  --query 'sort_by(imageDetails, &imagePushedAt)[*].{Tag:imageTags[0],Pushed:imagePushedAt}' \
  --output table
```

### Pin App Runner to a specific SHA tag

```bash
ROLLBACK_TAG="abc1234"   # replace with the tag from the table above

# Update the service to use the specific image tag
aws apprunner update-service \
  --service-arn $SERVICE_ARN \
  --region $AWS_REGION \
  --source-configuration "{
    \"ImageRepository\": {
      \"ImageIdentifier\": \"${ECR_REGISTRY}/${ECR_REPO}:${ROLLBACK_TAG}\",
      \"ImageConfiguration\": {
        \"Port\": \"8080\",
        \"RuntimeEnvironmentVariables\": {
          \"HF_MODEL_ID\":   \"${HF_MODEL_ID}\",
          \"HF_TOKEN\":      \"${HF_TOKEN}\",
          \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
          \"WANDB_PROJECT\": \"${WANDB_PROJECT}\"
        }
      },
      \"ImageRepositoryType\": \"ECR\"
    },
    \"AutoDeploymentsEnabled\": false
  }"
```

### Trigger the rollback deployment

```bash
aws apprunner start-deployment \
  --service-arn $SERVICE_ARN \
  --region $AWS_REGION
```

### Confirm rollback is running

```bash
aws apprunner describe-service \
  --service-arn $SERVICE_ARN \
  --region $AWS_REGION \
  --query 'Service.Status' \
  --output text
# Expected: RUNNING
```

---

## 10. Teardown / cleanup

> These commands are **destructive and irreversible**. Confirm `$SERVICE_ARN`
> and `$ECR_REPO` before running.

### Delete the App Runner service

```bash
aws apprunner delete-service \
  --service-arn $SERVICE_ARN \
  --region $AWS_REGION
# Deletion is async — the service transitions to DELETED within a few minutes.
```

### Delete the ECR repository and all images

```bash
aws ecr delete-repository \
  --repository-name $ECR_REPO \
  --region $AWS_REGION \
  --force
# --force deletes all images inside the repository first.
```

### Delete the IAM deployment user

```bash
# 1. Delete the access key (list keys first to find the KeyId)
aws iam list-access-keys --user-name flan-t5-deployer \
  --query 'AccessKeyMetadata[*].AccessKeyId' --output text | \
  xargs -I{} aws iam delete-access-key \
    --user-name flan-t5-deployer \
    --access-key-id {}

# 2. Remove the inline policy
aws iam delete-user-policy \
  --user-name flan-t5-deployer \
  --policy-name flan-t5-deploy-policy

# 3. Delete the user
aws iam delete-user --user-name flan-t5-deployer
```

---

*Maintained alongside `.github/workflows/deploy.yml` — keep `SERVICE_ARN`, port (`8080`), and env var names in sync.*
