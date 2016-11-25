import boto3
import argparse

bucket = boto3.resource('s3').Bucket('xcite-simulations')

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
args = parser.parse_args()

for item in bucket.objects.all():
    if not item.key.startswith(args.name):
        continue
    if item.key.endswith('converted-to.pdf') or item.key.endswith('.eps'):
        continue
    if 'report' in item.key and not item.key.endswith('.pdf'):
        continue
    print('https://s3-us-west-2.amazonaws.com/xcite-simulations/{}'.format(item.key))

