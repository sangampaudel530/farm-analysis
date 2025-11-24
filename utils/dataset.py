from roboflow import Roboflow

def get_data():
    """
    This function will:
    - Download the dataset from Roboflow.
    - Store it in the local directory 'data'.
    - Return the dataset object.

    Make sure the directory exists and the API key is configured correctly before calling this function.
    """

    rf = Roboflow(api_key="your_api_key_here") # Replace with your actual API key
    project = rf.workspace("workspace").project("project") # Replace with your actual workspace and project names
    version = project.version(15)
    dataset = version.download("png-mask-semantic") # Download in PNG format with semantic masks


# if __name__ == "__main__":
#     get_data()