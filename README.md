# Code-Catalyst

# A chatbot for solving coding problems. 

Hello guys, this is a chatbot I created using LLaMa2. I fine-tuned it on a dataset called CoNaLa which has gives different queries and have coding responses to it. I created the web application using FLask. I used HTML, CSS, and JavaScript. 

The "Templates" folder holds the HTML pages for the application and "static" folder has the different logos for it. 
The "app.py" file contains the backend of the project and it serves different purposes:

* It uses SQLAlchemy to store the data related to users
* It loads the model in Float16 to reduce the memory usage for quick inference.
* It communicates with frontend for taking user queries and then giving the answer from the model.
* It also has the ability to remember the chat history to keep track of the topics discussed.

  The "test.ipynb" shows the preprocessing for the dataset
  The "asd.py" file shows the configuration I used for fine-tuning the dataset.

  I hope you guys like the project. Although I am uploading it late, I had the idea before ChatGPT arrived, and I had started working on it in mid-2023.

  If you guys find any problems or bugs, please feel free to reach out to meharoo261 @ gmail.com. I would appreciate any improvements, suggestions, or questions. 
