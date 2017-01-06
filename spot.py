import json
import statistics
import subprocess
import os
import hashlib

import arrow

regions = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "ca-central-1",
    "ap-south-1",
    "ap-northeast-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-northeast-1",
    "eu-central-1",
    "eu-west-1",
    "eu-west-2",
    "sa-east-1"
]

instance_types = [
    "x1.32xlarge",
    "x1.16xlarge",
    "m4.16xlarge",
    "r4.16xlarge",
]


results = {}
CACHE_DIR = 'spot-cache'
os.makedirs(CACHE_DIR, exist_ok=True)

for region in regions:
    print('Checking {}'.format(region))
    for instance_type in instance_types:
        command = [
            "aws",
            "ec2",
            "describe-spot-price-history",
            "--start-time", "2017-01-01T00:00:00Z",
            "--instance-types", instance_type,
            "--product-description", "Linux/UNIX (Amazon VPC)",
            "--region", region
        ]

        key = hashlib.md5(json.dumps(command).encode('utf-8')).hexdigest()
        path = os.path.join(CACHE_DIR, key + '.json')
        if not os.path.exists(path):
            json_str = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
            open(path, 'w').write(json_str)
        data = json.load(open(path))
            
     
        # average per zone
        for record in data['SpotPriceHistory']:
            # record['Timestamp'] = arrow.get(record['Timestamp'])
            key = (record['AvailabilityZone'], record['InstanceType'])
            price = float(record['SpotPrice'])
            if instance_type == 'x1.32xlarge':
                price /= 2
            results.setdefault(key, []).append(price)

results = {k: statistics.mean(v) for k, v in results.items()}
results = sorted(results.items(), key=lambda t: t[1])
print()
print('Results')
print()
for zone, price in results:
    print('\t', zone, price)
