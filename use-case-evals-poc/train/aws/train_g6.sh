#!/usr/bin/env bash
# Backup trainer on AWS once the G-instance quota is approved (request 5201f731…, us-west-2, 8 vCPU).
# Launches one g6.xlarge (L4 24 GB) that trains with Unsloth, uploads the merged model to S3 and
# terminates itself. Hard stop after MAX_MIN minutes even if training hangs.
#
#   GH_S3=s3://<bucket>/gradientharbor GH_PROFILE=<instance-profile-with-s3> train/aws/train_g6.sh <round>
# Then locally: aws s3 sync $GH_S3/r<round>/merged models/ft-e4b-r<round>-hf
#               uv run mlx_lm.convert --hf-path models/ft-e4b-r<round>-hf --mlx-path models/ft-e4b-r<round> -q
set -euo pipefail
cd "$(dirname "$0")/.."
ROUND=${1:?round}; S3=${GH_S3:?set GH_S3}; PROFILE=${GH_PROFILE:?set GH_PROFILE (IAM instance profile with S3 access)}
REGION=${AWS_REGION:-us-west-2}; TYPE=${GH_INSTANCE:-g6.xlarge}; MAX_MIN=${MAX_MIN:-150}
AMI=$(aws ssm get-parameter --region "$REGION" --query Parameter.Value --output text \
  --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id)
aws s3 cp --recursive "data/r$ROUND" "$S3/r$ROUND/data"
aws s3 cp colab_unsloth.py "$S3/r$ROUND/train.py"
USERDATA=$(cat <<UD
#!/bin/bash
shutdown -h +$MAX_MIN
cd /opt && aws s3 cp --recursive $S3/r$ROUND/data data && aws s3 cp $S3/r$ROUND/train.py train.py
source /opt/pytorch/bin/activate 2>/dev/null || true
pip install -q unsloth && python train.py --data data --out merged > train.log 2>&1
aws s3 cp train.log $S3/r$ROUND/train.log; aws s3 cp --recursive merged $S3/r$ROUND/merged
shutdown -h now
UD
)
aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" \
  --iam-instance-profile Name="$PROFILE" --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=gradientharbor-train},{Key=project,Value=gradientharbor}]' \
  --user-data "$USERDATA" --query 'Instances[0].InstanceId' --output text
echo "instance terminates itself after training or $MAX_MIN min; watch $S3/r$ROUND/"
