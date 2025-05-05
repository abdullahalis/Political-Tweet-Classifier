FROM public.ecr.aws/lambda/python:3.9

ENV NLTK_DATA=/nltk_data

# Copy function code and model into container
COPY lambda_function.py model.pkl tfidf.pkl ./

# Install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt -t .

# Copy NLTK data
COPY nltk_data /nltk_data

# Set the Lambda function handler
CMD ["lambda_function.lambda_handler"]
