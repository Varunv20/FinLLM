import kagglehub

# Download latest version
path = kagglehub.dataset_download("ramssvimala/earning-call-transcripts")

print("Path to dataset files:", path)