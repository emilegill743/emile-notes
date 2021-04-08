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
    - [Sharing files through a website](#sharing-files-through-a-website)
    - [Case Study: Generating a Report Repository](#case-study-generating-a-report-repository)
  - [SNS Topics](#sns-topics)
    - [Creating an SNS Topic](#creating-an-sns-topic)
    - [Listing Topics](#listing-topics)
    - [Deleting Topics](#deleting-topics)
  - [SNS Subscriptions](#sns-subscriptions)
    - [Listing Subscriptions](#listing-subscriptions)
    - [Deleting Subscriptions](#deleting-subscriptions)
    - [Deleting multiple subscriptions](#deleting-multiple-subscriptions)
  - [Sending Messages](#sending-messages)
    - [Case Study: Building a notification system](#case-study-building-a-notification-system)
  - [Computer Vision: AWS Rekognition](#computer-vision-aws-rekognition)
  - [NLP: AWS Translate, AWS Comprehend](#nlp-aws-translate-aws-comprehend)
    - [Case Study: Detecting sentiment about e-scooter blocking the sidewalk](#case-study-detecting-sentiment-about-e-scooter-blocking-the-sidewalk)
- [Initialise Boto3 Clients](#initialise-boto3-clients)
- [Translate all descriptions into English](#translate-all-descriptions-into-english)
- [Detect text sentiment](#detect-text-sentiment)
- [Detect scooter in image](#detect-scooter-in-image)
- [Select only rows where there was a scooter image and negative sentiment](#select-only-rows-where-there-was-a-scooter-image-and-negative-sentiment)

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

### Sharing files through a website

Converting a DataFrame to html:

```python
df.to_html('table_agg.html',
           render_links=True,
           columns['service_name', 'request_count', 'info_link'],
           borders=0)
```

Upload an HTML file to S3:

```python
s.upload_file(
  Filename='./table_agg.html',
  Bucket='datacamp-website',
  Key='table.html',
  ExtraArgs = {
    'ContentType': 'text/html',
    'ACL': 'public-read'}
)
```

Accessing HTML file:

`https://{bucket}.s3.amazonaws.com/{key}`

`https://datacamp-website.s3.amazonaws.com/table.html`

Uploading an image file:

```python
s3.upload_file(
  Filename='./plot_image.png',
  Bucket='datacamp-website',
  Key='plot_image.png',
  ExtraArgs = {
    'ContentType': 'image/png',
    'ACL': 'public-read'}
)
```

IANA Media Types:

Full list of IANA media types can be found at, http://www.iana.org/assignments/media-types/media-types.xhtml.

Common types include: JSON, `application/json`; PNG, `image/png`; PDF, `application/pdf`; CSV, `text/csv`.

Generating an index page:

```python
# List the gid-reports bucket objects starting with 2019/
r = s3.list_objects(BUcket='gid-reports', Prefix='2019/')

# Convert the response contents to DataFrame
objects_df = pd.DataFrame(r['Contents'])

# Create column "Link" that contains website url + key
base_url = 'http://datacamp-website.s3.amazonaws.com/"
objects_df['Link'] = base_url + objects_df['Key']

# Write DataFrame to html
objects_df.to_html('report_listing.html',
                   columns=['Link', 'LastModified', 'Size'],
                   render_links=True)

# Upload HTML file to S3
s3.upload_file(
  Filename='./report_listing.html',
  Bucket='datacamp-website',
  Key='index.html',
  ExtraArgs = {
    'ContentType': 'text/html',
    'ACL': 'public-read'}
)
```

### Case Study: Generating a Report Repository

1) Prepare the data

- Download files for the month from the raw data bucket
  
```python
# Create list to hold our DataFrames
df_list = []

# Request the list of CSVs from S3 with prefix; Get contents
response = s3.list_objects(
  Bucket='gid-requests',
  Prefix='2019_jan')

# Get response contents
request_files = response['Contents]
```

- Concatenate them into one DataFrame

```python
# Iterate over each object
for file in request_files:
   obj = s3.get_object(Bucket='gid-requests', Key=file['Key'])

   # Read it as DataFrame
   obj_df = pd.read_csv(obj['Body'])

   # Append DataFrame to list
   df_list.append(obj_df)

# Concatenate all dfs in list
df = pd.concat(df_list)
```
- Create a aggregated DataFrame

1) Create the Report

- Write the DataFrame to CSV and HTML

```python
# Write agg_df to a CSV and HTML file with no border
agg_df.to_csv('./jan_final_report.csv')
agg_df.to_html('./jan_final_report.html', border=0)
```

- Generate a Bokeh plot, save as HTML

1) Upload report to shareable website

- Create `gid-reports` bucket
- Upload all three files for month to S3

```python
# Upload Aggregated CSV to S3
s3.upload_file(Filename='./jan_final_report.csv',
               Key='2019/jan/final_report.csv',
               Bucket='gid-reports',
               ExtraArgs = {'ACL': 'public-read'})

# Upload HTML table to S3
s3.upload_file(Filename='./jan_final_report.html',
               Key='2019/jan/final_report.html',
               Bucket='gid-reports',
               ExtraArgs = {
                 'ContentType': 'text/html',
                 'ACL': 'public-read'})

# Upload Aggregated Chart to S3
s3.upload_file(Filename='./jan_final_chart.html',
               Key='2019/jan/final_chart.html',
               Bucket='gid-reports',
               ExtraArgs = {
                 'ContentType': 'text/html',
                 'ACL': 'public-read'})

```
- Generate an index.html file that lists all the files

```python
# List the gid-reports bucket objects starting with 2019/
r = s3.list_objects(Bucket='gid-reports', Prefix='2019/')

# Convert the response contents to DataFrame
objects_df = pd.DataFrame(r['Contents'])

# Create a column "Link" that contains website url + key
base_url = "https://gid-reports.s3.amazonaws.com/"
objects_df['Link'] = base_url + objects_df['Key']

# Write DataFrame to html
objects_df.to_html('report_listing.html',
                   columns=['Link', 'LastModified', 'Size'],
                   render_links=True)

# Upload the file to gid-reports bucket root
s3.upload_file(Filename='./report_listing.html',
               Key='index.html',
               Bucket='gid-reports',
               ExtraArgs = {
                 'ContentType': 'text/html',
                 'ACL': 'public-read'})
```
- Get website URL

`http://gid-reports.s3.amazonaws.com/index.html`

## SNS Topics

**SNS** - Simple Notification Service

Publisher --> SNS Topic --> SNS Subscriber (via Email or Text)

Each SNS has a unique ARN (Amazon Resource Name).
Each Subscription has a unique ID.

### Creating an SNS Topic

```python
sns = boto3.client('sns',
                   region_name='us_east-1',
                   aws_access_key_id=AWS_KEY_ID,
                   aws_secret_access_key=AWS SECRET)

response = sns.create_topic(Name='city_alerts')

topic_arn = response['TopicArn]
```

- `create_topic()` method is idempotent, so won't recreate a topic if it already exists

### Listing Topics

```python
sns.list_topics()
```

### Deleting Topics

```python
sns.delete_topic(TopicArn='arn:aws:sns:us-east-1:320333787981:city_alerts')
```

```python
# Get the current list of topics
topics = sns.list_topics()['Topics']

for topic in topics:
  # For each topic, if it is not marked critical, delete it
  if "critical" not in topic['TopicArn']:
    sns.delete_topic(TopicArn=topic['TopicArn'])
    
# Print the list of remaining critical topics
print(sns.list_topics()['Topics'])
```

## SNS Subscriptions

Each subscription has a unique ID, Endpoint (phone number or email address where message sent), Status (Confirmed or pending confirmation) and Protocol (Email or SMS).

```python
sns = boto3.client('sns',
                   region_name='us_east-1',
                   aws_access_key_id=AWS_KEY_ID,
                   aws_secret_access_key=AWS SECRET)

response = sns.subscribe(
  TopicArn = 'arn:aws:sns:us-east-1:320333787981:city_alerts',
  Protocol = 'SMS',
  Endpoint = '+13125551123')
```

Subscriptions automatically confirmed for SMS, but will be 'Pending Confirmation' for email until user confirms.

### Listing Subscriptions

```python
sns.list_subscriptions_by_topic(
  TopicArn='arn:aws:sns:us-east-1:320333787981:city_alerts')
```

```python
sns.list_subscriptions()['Subscriptions']
```

### Deleting Subscriptions

```python
sns.unsubscribe(
    SubscriptionArn='arn:aws:sns:us-east-1:320333787981:city_alerts:9f2dad1d-844')
```

### Deleting multiple subscriptions

```python
response = sns.list_subscriptions_by_topic(
  TopicArn='arn:aws:sns:us-east-1:320333787981:city_alerts')

subs = response['Subscriptions']

for sub in subs:
  if sub['Protocol'] = 'sms':
    sns.unsubscribe(sub['SubscriptionArn'])
```

## Sending Messages

Publishing to a topic:

```python
response = sns.publish(
  TopicArn='arn:aws:sns:us-east-1:320333787981:city_alerts',
  Message='Body of SMS or e-mail',
  Subject='Subject Line for Email'
)
```

Sending custom messages:

```python
num_of_reports = 137

response = client.publish(
  TopicArn='arn:aws:sns:us-east-1:320333787981:city_alerts',
  Message='There are {} reports outstanding'.format(num_of_reports),
  Subject='Subject Line for Email'
)
```

Sending a single SMS without Topic or Subscriber:

```python
response = sns.publish(
  PhoneNumber='+13121233211',
  Message='Body text of SMS or e-mail'
)
```

- Good for one  off, but not good long term practice

### Case Study: Building a notification system

1) Topic Set Up

- Create topics for each service

```python
sns = boto3.client('sns',
                   region_name='us-east-1',
                   aws_access_key_id=AWS_KEY_ID,
                   aws_secret_access_key=AWS_SECRET)

trash_arn = sns.create_topic(Name='trash_notifications')['TopicArn']
streets_arn = sns.create_topic(Name='streets_notifications')['TopicArn']
```

- Download the contact list CSV

```python
contacts = pd.read_csv('http://gid-staging.s3.amazonaws.com/contacts.csv')
```

- Subscribe the contacts to their respective

```python
def subscribe_user(user_row):
   if user_row['Department'] == 'trash':
      sns.subscribe(TopicArn=trash_arn, Protocol='sms', Endpoint=str(user_row['Phone']))
      sns.subscribe(TopicArn=trash_arn, Protocol='email', Endpoint=str(user_row['Email']))
   else:
      sns.subscribe(TopicArn=streets_arn, Protocol='sms', Endpoint=str(user_row['Phone']))
      sns.subscribe(TopicArn=streets_arn, Protocol='email', Endpoint=str(user_row['Email']))

contacts.apply(subscribe_user, axis=1)
```

2) Get aggregated numbers

- Download monthly report

```python
df = pd.read_csv('http://gid-reports.s3.amazonaws.com/2019/feb/final_report.csv')
```

- Get count of potholes and illegal dumpings

```python
df.set_index('service_name', inplace=True)

trash_violations_count = df.at['Illegal Dumping', 'count']
streets_violations_count = df.at['Pothole', 'count']
```

3) Send alerts

- If potholes exceeds 100, send alert
- If illegal dumping exceeds 30, send alert

```python
if trash_violations_count > 100:

   message = "Trash violations count is now {}".format(trash_violations_count)

   sns.publish(TopicArn=trash_arn,
               Message=message,
               Subject="Trash Alert")

if streets_violations_count > 30:

   message = "Streets violations count is now {}".format(streets_violations_count)

   sns.publish(TopicArn=streets_arn,
               Message=message,
               Subject="Streets Alert")
```

## Computer Vision: AWS Rekognition

Boto3 follows the same pattern for all AWS services.

Rekognition is a computer vision API by AWS. Uses include: detecting objects in an image and extracting text from images.

Upload an image to S3:

```python
# Initialise S3 Client
s3 = boto3.client(
  's3', region_name='us-east-1',
  aws_access_key_id=AWS_KEY_ID,
  aws_secret_access_key=AWS_SECRET
)

# Upload file
s3.upload_file(
  Filename='report.jpg',
  Key='report.jpg',
  Bucket='datacamp-img')
  ```

Object detection:

```python
# Construct Rekogition Client
rekog = boto3.client(
  'rekognition',
  region_name='us-east-1',
  aws_secret_key_id=AWS_KEY_ID,
  aws_secret_access_key=AWS_SECRET)

# Call detect_labels method
response = rekog.detect_labels(
   Image={'S3Object': {
             'Bucket': 'datacamp-img',
             'Name': 'report.jpg'}
         },
   MaxLabels=10,
   MinConfidence=95
)
```

Text Detection:

```python
response = rekog.detect_text(
   Image={'S3Object': {
              'Bucket': 'datacamp-img',
              'Name': 'report.jpg'}
          }
)
```

Returns "line" (rows of text) and "word" detections (individual words).

## NLP: AWS Translate, AWS Comprehend

Translating text:

```python
# Initialise client
translate = boto3.client('translate',
                         region_name='us-east-1',
                         aws_access_key_id=AWS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET)

# Translate Text
response = translate.translate_text(
              Text='Hello, how are you?',
              SourceLanguageCode='auto',
              TargetLanguageCode='es')

translated_text = response['TranslatedText']
```

Detecting Language:

```python
# Initialise client
comprehend = boto3.client('comprehend',
                          region_name='us-east-1',
                          aws_access_key_id=AWS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET)

# Detect dominant language
response = comprehend.detect_dominant_language(
  Text="Hay basura por todas partes a lo largo de la carretera."
)
```

Understanding Sentiment:

```python
# Detect text sentiment
response = comprehend.detect_sentiment(
   Text="DataCamp students are amazing.",
   LanguageCode='en')

sentiment = response['Sentiment]
```

### Case Study: Detecting sentiment about e-scooter blocking the sidewalk

```python
# Initialise Boto3 Clients
rekog = boto3.client('rekognition',
                     region_name='us-east-1',
                     aws_access_key_id=AWS_KEY_ID,
                     aws_secret_access_key=AWS_SECRET)

comprehend = boto3.client('comprehend',
                          region_name='us-east-1',
                          aws_access_key_id=AWS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET)

translate = boto3.client('translate',
                         region_name='us-east-1',
                         aws_access_key_id=AWS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET)

# Translate all descriptions into English
for index, row in df.iterrows():
   desc = df.loc[index, 'public_description']
   if desc != '':
      resp = translate.translate_text(
                Text=desc,
                SourceLanguageCode='auto',
                TargetLanguageCode='en')
      df.loc[index, 'public_descriprion'] = resp['TranslatedText']

# Detect text sentiment
for index, row in df.iterrows():
   desc = df.loc[index, 'public_descriprion']
   if desc != '':
      resp = comprehend.detect_sentiment(
                Text=desc,
                LanguageCode='en')
      df.loc[index, 'sentiment'] = resp['Sentiment']

# Detect scooter in image
df['img_scooter'] = 0
for index, row in df.iterrows():
   image = df.loc[index, 'image']
   response = rekog.detect_labels(
                 Image={'S3Object': 
                          {'Bucket': 'gid-images',
                           'Name': image}})
   for label in response['Labels']:
      if label['Name'] == 'Scooter':
         df.loc[index, 'img_scooter'] = 1
         break

# Select only rows where there was a scooter image and negative sentiment
pickups = df[((df.img_scooter == 1)
              & (df.sentiment == 'NEGATIVE'))]

num_pickups = len(pickups)


