#!/bin/bash -e



#aws ec2 describe-spot-price-history --start-time=2017-01-01T00:00:00Z --instance-types=x1.32xlarge --product-description='Linux/UNIX (Amazon VPC)' --region=us-west-2
#aws ec2 describe-spot-price-history --start-time=2017-01-01T00:00:00Z --instance-types=x1.32xlarge --product-description='Linux/UNIX (Amazon VPC)' --region=us-east-1
aws ec2 describe-spot-price-history --start-time=2017-01-01T00:00:00Z --instance-types=x1.32xlarge --product-description='Linux/UNIX (Amazon VPC)' --region=ca-central-1
