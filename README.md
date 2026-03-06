\# Intelligent Seed Prediction and Recommendation System 



A \*\*Machine Learning-based web application\*\* that recommends suitable crops based on soil nutrients, atmospheric conditions, rainfall, and geographical data.



This system helps farmers and agricultural planners choose the \*\*best crop for specific environmental conditions\*\*.



---







\## Features



\* Crop recommendation using Machine Learning

\* Atmospheric parameter-based prediction

\* District and altitude-based crop suggestions

\* Data visualization using Matplotlib

\* Interactive web interface using Flask



---



\## Tech Stack



\* Python

\* Flask

\* Scikit-learn

\* NumPy

\* Matplotlib

\* HTML / CSS

\* Docker



---



\## Project Structure



```

backend.py                # Flask backend application

Training code.py          # Machine learning model training

model\_and\_scaler.pkl      # Trained ML model

Crop\_recommendation.csv   # Dataset used for training

templates/                # HTML pages

static/                   # Images and static assets

requirements.txt          # Python dependencies

Dockerfile                # Docker container configuration

```



---



\## Installation (Run Locally)



1\. Clone the repository



```

git clone https://github.com/gopiO-O/Intelligent-seed-prediction-and-recommendation-system.git

```



2\. Navigate to project folder



```

cd Intelligent-seed-prediction-and-recommendation-system

```



3\. Install dependencies



```

pip install -r requirements.txt

```



4\. Run the application



```

python backend.py

```



Open in browser:



```

http://localhost:5000

```



---



\## Run Using Docker



Build Docker image



```

docker build -t intelligent-seed-prediction-system .

```



Run container



```

docker run -p 5000:5000 intelligent-seed-prediction-system

```



---



\## Machine Learning Model



The system uses a \*\*Decision Tree based crop recommendation model\*\* trained on agricultural data including:



\* Nitrogen (N)

\* Phosphorus (P)

\* Potassium (K)

\* Temperature

\* Humidity

\* pH level

\* Rainfall



---






\* Deploy on AWS EC2

\* Real-time weather API

\* Farmer dashboard

\* Soil sensor integration




\## Author



\*\*Gopi Krishnan\*\*



GitHub:

https://github.com/gopiO-O



