# Introduction to AWS Boto in Python

- [Introduction to AWS Boto in Python](#introduction-to-aws-boto-in-python)
  - [Intro to AWS and Boto3](#intro-to-aws-and-boto3)
    - [Creating an IAM User](#creating-an-iam-user)
    - [AWS Services](#aws-services)
  - [Diving into buckets](#diving-into-buckets)
    - [What can we do with buckets using Boto3?](#what-can-we-do-with-buckets-using-boto3)
  - [Uploading and Retrieving Files](#uploading-and-retrieving-files)
    - [Upload files](#upload-files)
    - [List objects in a bucket](#list-objects-in-a-bucket)
    - [Get object metadata](#get-object-metadata)
    - [Download file](#download-file)
    - [Delete objects](#delete-objects)
  - [Sharing Files Securely](#sharing-files-securely)
    - [ACLs](#acls)
    - [Accessing public objects](#accessing-public-objects)
    - [Downloading a private file](#downloading-a-private-file)
    - [Pre-signed URLs](#pre-signed-urls)

## Intro to AWS and Boto3

AWS provide storage, compute and alerts services that we can leverage in data projects. AWS services are granular, meaning they can work together, or on their own.

To interact with AWS services using Python we can use the `Boto3` library:

```python
import boto3

s3 = boto3.client(
        's3',
        region_name='us-east-1',
        aws_access_key_id=AWS_KEY_ID,
        aws_secret_access_key=AWS_SECRET)

response = s3.list_buckets()
```

### Creating an IAM User

> IAM - **Identity Access Management**

- IAM sub-users allow us to control access to AWS services within our account.

- Authenticated with Key/Secret.

To create an IAM role, we can search for IAM in the AWS Management Console. Selecting `Users` followed by `Add User` allows us to input a User name and select `Programmatic Access`, to allow us to use a access key/secret to connect via Boto3.

We can select **'Attach existing policies directly'** and search for the policy we wish to attach, e.g. `s3fullaccess`.

We will then be provided with an **Access key ID** and a **Secret access key** that we can use to programmatically interact with AWS.

### AWS Services

**S3 (Simple Storage Service)** - Lets us store files in the cloud.

**SNS (Simple Notification Service)** - Lets us send emails and texts to alert subscribers based on events and conditions in our data pipelines.

**Comprehend** - Performs sentiment analysis on blocks of text.

**Recognition** - Extracts text from images.

---

## Diving into buckets

S3 allows us to place files in the cloud, accessible anywhere by a url.

**Buckets** - Analogous to Desktop folders

- Own permission policy
- Website storage (statis website hosting)
- Generate logs about activity

**Objects** - Analogous to files

### What can we do with buckets using Boto3?

- Creating a Bucket

```python
import boto3

# Create boto3 client
s3 = boto.client(
            's3',
            region_name='us-east-1',
            aws_access_key=AWS_KEY_ID,
            aws_secret_access_key=AWS_SECRET)

# Create bucket
bucket = s3.create_bucket(Bucket='gid-requests)
```

- List Buckets

```python
import boto3

s3 = boto.client(
            's3',
            region_name='us-east-1',
            aws_access_key=AWS_KEY_ID,
            aws_secret_access_key=AWS_SECRET)

# List buckets as dict
bucket_response = s3.list_buckets()
```

- Delete Bucket

```python
import boto3

s3 = boto.client(
            's3',
            region_name='us-east-1',
            aws_access_key=AWS_KEY_ID,
            aws_secret_access_key=AWS_SECRET)

# Delete bucket
response = s3.delete_bucket('gid-requests')
```

---

## Uploading and Retrieving Files

An object can be anything: image, video file, csv, log file etc.

Objects and buckets analogous to files and folders on a desktop.

|   Bucket  |   Object  |
|   ------  |   ------  |
| A bucket has a ***name*** |An object has a ***key*** |
| ***Name*** is a string | ***Name*** is full path from bucket root |
| ***Unique name*** in all of s3 | ***Unique key*** in bucket |
| Contains ***many*** objects | Can only be in ***one*** parent bucket |

<br>

### Upload files

```python
s3.upload_file(
    Filename='gid_requests_2019_01_01.csv', # local file path
    Bucket='gid-requests', # name of bucket we're uploading to
    Key='gid_requests_2019_01_01.csv') # name in s3 
```

### List objects in a bucket

```python
response = s3.list_objects(
                Bucket='gid-requests',
                MaxKeys=2, # limit response to n objects
                Prefix='gid_requests_2019_') # limit to objects starting with prefix
```

### Get object metadata

```python
response = s3.head_objects(
                Bucket='gid-requests',
                Key='gid_requests_2018_12_30.csv)
```

### Download file

```python
s3.download_file(
    Filename='gid_requests_downed.csv', # local file path to download to
    Bucket='gid-requests',
    Key='gid_requests_2018_12_30.csv')
```

### Delete objects

```python
s3.delete_object(
    Bucket='gid-requests',
    Key='gid_requests_2018_12_30.csv')
```

---

## Sharing Files Securely

AWS defaults to denying permission, so we must explicitly grant access in order to access objects/buckets.

AWS Permissions Systems:

- IAM - Attach a policy to specific users to control access to services, buckets and objects.
- Bucket policy - Control buckets and objects within.
- ACL (Access Control Lists) - Set permissions on specific objects within a bucket.
- Presigned URL - Temporary access to an object.

### ACLs

Entities attached to objects in s3. Common types include: `'public-read'` and `'private'`

```python
# By default, when unspecified ACL is 'private'
s3.upload_file(
  Filename='potholes.csv', Bucket='gid-requests', Key='potholes.csv')

# Set ACL to 'public-read'
s3.put_object_acl(
  Bucket='gid-requests', Key='potholes.csv', ACL='public-read')
```

Setting ACLs on upload:

```python
s3.upload_file(
    Bucket='gid-requests',
    Filename='potholes.csv',
    Key='potholes.csv',
    ExtraArgs{'ACL':'public-read'})
```

### Accessing public objects

Publicly accessible s3 objects can be accessed using the url template:
```
https://{bucket}.s3.amazonaws.com/{key}
```

e.g. `https://gid-requests.s3.amazonaws.com/2019/potholes.csv`

Generating public objects URL:

```python
url = "https://{}.s3.amazonaws.com/{}".format(
  "gid-requests",
  "2019/potholes.csv")

df = pd.read_csv(url)
```

### Downloading a private file

```python
# Download file
s3.download_file(
  Filename='potholes_local.csv',
  Bucket='gid-staging',
  Key='2019/potholes_private.csv')

# Read from Disk
pd.read_csv('./potholes_local.csv')
```

In memory read of file:

```python
obj = s3.get_object(Bucket='gid-requests', Key='2019/potholes.csv')

pd.read_csv(obj['Body']) # Read StreamingBody object into Pandas
```

### Pre-signed URLs

Grant temporary access to s3 objects, with an expiring timeframe.

```python
# Generate Presigned URL

share_url = s3.generate_presigned_url(
  ClientMethod='get_object',
  ExpiresIn=3600, # grant access for one hour
  Params={'Bucket':'gid-requests', 'Key'='potholes.csv'}
)

pd.read_csv(share_url)
```

Load multiple files into one DataFrame:

```python
# Create list to hold our DataFrames
df_list = []


# Request the list of csv's from S3 with prefix; Get contents
response = s3.list_objects(
  Bucket='gid-requests', 
  Prefix='2019/')


# Get response contents
request_files = response['Contents']

# Iterate over each object
for file in request_files:
  obj = s3.get_object(Bucket='gid-requests', Key=file['Key'])

  #Read it as DataFrame
  obj_df = pd.read_csv(obj['Body'])

  # Append DataFrame to list
  df_list.append(obj_df)

# Concatenate all the DataFrames in the list
df = pd.concat(df_list)
```




















