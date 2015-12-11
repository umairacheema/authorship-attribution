#!/bin/bash
# Script to create AWS Elastic MapReduce Cluster
# Change the variables given below to configure
# Cluster.Fill in the empty variables before
# running this script.
# The script requires AWS CLI to be installed 
# and configured.
# IMPORTANT: AWS EMR cluster is a paid cloud service by Amazon
# It is essential to look at price for the region
# before creating a cluster
# It is also imperative to terminate the cluster after using 

CLUSTER_NAME="GutenbergEMR"
EMR_RELEASE_LABEL="emr-4.2.0"
APPLICATION_NAME="Spark"

#Key name from AWS account to be associated with cluster in order to 
#configure access to nodes
KEY_NAME=""

EC2_INSTANCE_TYPE="m3.2xlarge"
EC2_INSTANCE_COUNT=5

#Location of the bootstrap script
#Copy the install-anaconda script to s3 and give its path
# e.g s3://s3bucketname/install-anaconda
BOOTSTRAP_SCRIPT_PATH_S3=""
SPARK_CONFIG_FILE_PATH="file://./spark-conf.json"

#Location of debug logs on s3
#s3://s3bucketname/emr-logs
DEBUG_LOG_PATH_S3=""

export CLUSTER_ID=`aws emr create-cluster --name $CLUSTER_NAME  --release-label $EMR_RELEASE_LABEL --applications Name=$APPLICATION_NAME --ec2-attributes KeyName=$KEY_NAME  --instance-type $EC2_INSTANCE_TYPE --instance-count $EC2_INSTANCE_COUNT --use-default-roles --configurations $SPARK_CONFIG_FILE_PATH --enable-debugging --log-uri $DEBUG_LOG_PATH_S3 --bootstrap-actions Path=$BOOTSTRAP_SCRIPT_PATH_S3,Name=custom_bootstrap  --query 'ClusterId' --output text` && echo $CLUSTER_ID
