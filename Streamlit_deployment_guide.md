# Streamlit Deployment Guide

This guide explains how to deploy your Data Analytics application to Streamlit Cloud.

## Prerequisites

Before deployment, ensure you have the following:

1. A GitHub account
2. The complete project code pushed to a GitHub repository
3. A free Streamlit Cloud account

## Preparing Your Project for Deployment

1. **Organize your project structure** as shown in the provided project structure
2. **Create a requirements.txt file** with all necessary dependencies
3. **Ensure all file paths are relative** and work both locally and in the cloud
4. **Include a .streamlit folder** with a config.toml file for styling (optional)

## Local Testing

Before deploying to Streamlit Cloud, test your application locally:

```bash
# Navigate to your project directory
cd bakery-analytics

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py
```

Verify that all features work correctly and that data loading functions as expected.

## Deployment to Streamlit Cloud

Follow these steps to deploy your application:

1. **Push your code to GitHub**

```bash
git init
git add .
git commit -m "Initial commit for Streamlit deployment"
git branch -M main
git remote add origin https://github.com/yourusername/bakery-analytics.git
git push -u origin main
```

2. **Sign up for Streamlit Cloud**

   Visit [Streamlit Cloud](https://streamlit.io/cloud) and sign up for a free account.

3. **Deploy your app**

   - Click "New app" in the Streamlit Cloud dashboard
   - Connect your GitHub repository
   - Set the Python file path to `app/app.py`
   - Configure advanced settings if needed (Python version, requirements path)
   - Click "Deploy"

4. **Monitor deployment**

   Streamlit Cloud will build and deploy your app. Check the logs for any errors.

5. **Share your app**

   Once deployment is complete, you'll receive a public URL for your application.

## Setting Up Automatic Updates

Streamlit Cloud can automatically update your app when you push changes to GitHub:

1. In your Streamlit Cloud dashboard, select your app
2. Go to "Settings" > "Advanced"
3. Enable "Reboot app when GitHub repo has changes"

## Environmental Variables and Secrets

For sensitive information (if needed):

1. In your Streamlit Cloud dashboard, select your app
2. Go to "Settings" > "Secrets"
3. Add your secrets in the form of key-value pairs

Access these secrets in your code using:

```python
import streamlit as st

# Access secrets
secret_value = st.secrets["your_secret_key"]
```

## Troubleshooting Common Issues

- **App fails to build**: Check requirements.txt for compatibility issues
- **Data file not found**: Ensure all file paths are correct and data files are included in the repository
- **Memory issues**: Optimize your code to reduce memory usage or upgrade your Streamlit Cloud plan

## Sharing Your Dashboard

Once deployed, you can share your dashboard with:

1. **Direct link sharing**: Share the public URL
2. **Embed in websites**: Use an iframe to embed the app
3. **Password protection**: Set up authentication in Streamlit Cloud settings

## Next Steps

After deployment, consider:

- Setting up a CI/CD pipeline for automated testing
- Adding authentication for secure access
- Creating a monitoring system to track performance metrics
- Gathering user feedback to improve the dashboard

By following this guide, your Data Analytics dashboard will be available online for stakeholders to access from anywhere.