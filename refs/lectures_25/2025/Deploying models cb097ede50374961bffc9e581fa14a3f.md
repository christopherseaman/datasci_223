# Deploying models

### Deploying Machine Learning Models

### Non-Production Deployments

- **Streamlit**: Rapidly develop and share data apps directly from Python scripts. Ideal for prototyping, data analysis, and sharing insights.
- **HuggingFace**: Host and share ML models and demos with the community. Supports various frameworks and offers interactive API testing.

### Production Deployments

- **Cloud**: Deploy scalable ML models using services like AWS/GCP/Azure for model training and deployment. Offers integration with other cloud services for data processing and storage.
- **Self-hosted:** Host your own models using Python + Flask. You will have to define an HTML front-end and API endpoints for it to call to run predictions through your model. More complicated than Streamlit or HuggingFace, but only differently complicated from deploying to the cloud.

### Additional Considerations

- **Model Monitoring**: Track performance, usage, and health of deployed models. Essential for maintaining accuracy over time.
- **Security**: Implement authentication, encryption, and access controls to protect your models and data.
- **Versioning**: Manage model versions to rollback or update deployed models efficiently.

# Deploying models with Streamlit

## 🌐 Why Streamlit?

- **Empowering Data Scientists:** Streamlit is a game-changer for data scientists looking to turn data scripts into shareable web apps with minimal hassle. Its simplicity and efficiency make it an indispensable tool in the modern data science toolkit.
- **Ease of Use:** Streamlit's intuitive API and straightforward approach allow for quick creation of interactive dashboards. This simplicity enables data scientists to focus on the data and models rather than web development intricacies.
- **Rapid Prototyping:** With Streamlit, you can rapidly prototype and iterate on your data applications, making it easier to explore, explain, and share your work with stakeholders.
- **Seamless Integration:** Streamlit seamlessly integrates with major data science libraries like Pandas, NumPy, Matplotlib, and Plotly, allowing you to leverage existing Python skills and resources.

## 🚀 Getting Started with Streamlit

```bash
pip install streamlit
```

## ## 🧑‍💻 Creating Your First Streamlit App

1. **Write Your Streamlit Script**
    
    Start by creating a Python script (e.g., `app.py`). Import Streamlit and other necessary libraries. Use Streamlit commands to add elements to your app.
    
    ```python
    import streamlit as st
    import pandas as pd
    
    # Title of the app
    st.title('My First Streamlit App')
    
    # Displaying a dataframe
    df = pd.DataFrame({'Column 1': [1, 2, 3], 'Column 2': [4, 5, 6]})
    st.write(df)
    ```
    
2. **Run Your App**
    
    Launch your app from the command line by navigating to your script's directory and running:
    
    ```bash
    streamlit run app.py
    ```
    
    Streamlit will start a local server, and your default web browser will open with your new app.
    

## 🌟 Deploying a Model with Streamlit

### `Hello, World!` to make sure you’re set up properly:

[Install Streamlit using command line - Streamlit Docs](https://docs.streamlit.io/get-started/installation/command-line)

### Deploying an existing model using Streamlit:

[Deploying Machine Learning Models with Python & Streamlit | 365 Data Science](https://365datascience.com/tutorials/machine-learning-tutorials/how-to-deploy-machine-learning-models-with-python-and-streamlit/)

### 🧰 Advanced Features

- **Caching:** Use `@st.cache` to cache data loading and processing, speeding up your apps.
- **Layouts and Containers:** Organize your app with sidebars, columns, and expanders for a polished look.
- **Interactive Widgets:** Integrate sliders, buttons, and other widgets for dynamic user interaction.
- **Plotting and Visualization:** Embed Matplotlib, Plotly, and other visualizations directly in your app.

## 🤗Hugging Face

HuggingFace is kind of like GitHub for models. They have amazing docs for sharing models and deploying them directly on Hugging Face or to various cloud providers. Under the hood you can use `streamlit` (or `gradio` or whatever else), but you don’t have to host the model yourself.

### Deploying models with Hugging Face Spaces

[Spaces Overview](https://huggingface.co/docs/hub/en/spaces-overview)

### Broader Hugging Face docs:

[Hugging Face - Documentation](https://huggingface.co/docs)

# Cloud

Building and deploying to AWS/GCP/Azure can be quite involved. I’ve written up a walkthrough of the process of deploying a toy, pre-built model to AWS here:

- [The Life of a Toy Model](https://docs.google.com/document/d/1KSxvVTV-STotoH0_EomhXQZuhekaU0uRl3Iz_7vI_z0/edit)