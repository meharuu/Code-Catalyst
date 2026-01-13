# Code-Catalyst

# A chatbot for solving coding problems. 

Hello guys, this is a chatbot I created using LLaMa2. I fine-tuned it on a dataset called CoNaLa, which gives different queries and has coding responses to them. I created the web application using Flask. I used HTML, CSS, and JavaScript. 

The "Templates" folder contains the HTML pages for the application, and the "static" folder holds the various logos used for it. 
The "app.py" file contains the backend of the project, and it serves different purposes:

* It uses SQLAlchemy to store the data related to users
* It loads the model in Float16 to reduce the memory usage for quick inference.
* It communicates with the frontend for taking user queries and then giving the answer from the model.
* It can also remember the chat history to keep track of the topics discussed.

  The "configuration.py" file shows the configuration I used for fine-tuning the dataset.

  I hope you guys like the project. Although I am uploading it late, I had the idea before ChatGPT arrived, and I had started working on it in mid-2023.

  If you guys find any problems or bugs, please feel free to reach out to meharoo261 @ gmail.com. I would appreciate any improvements, suggestions, or questions. 
