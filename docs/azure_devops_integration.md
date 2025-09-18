
# Azure DevOps Integration

This guide outlines how to integrate the MLOps solution accelerator repository with Azure DevOps for version control, task tracking, and CI/CD pipelines.

## Version Control

You can host this repository in **Azure Repos**. To do so:

1. Create a new repository in Azure DevOps under your project.
2. Clone the repository locally and add this code as your initial commit:
   ```bash
   git remote add origin <your-azure-repo-url>
   git branch -M main
   git push -u origin main
   ```

## Boards & Work Items

- Use **Azure Boards** to manage the project’s backlog, tasks, and user stories.
- Link commits and pull requests to Azure Boards work items by including the work item ID in commit messages (e.g., `#AB123`).
- Configure pull request policies to require code reviews before merging into the main branch.

## Pipelines (Optional)

Although deployment is out of scope for this version, you can set up **Azure Pipelines** for continuous integration to run unit tests and linting:

1. Create a `azure-pipelines.yml` in the repository root:
   ```yaml
   trigger:
     branches:
       include:
         - main

   jobs:
   - job: Build
     pool:
       vmImage: 'ubuntu-latest'
     steps:
     - task: UsePythonVersion@0
       inputs:
         versionSpec: '3.8'
     - script: |
         pip install -r requirements.txt
         # run unit tests if available
         # python -m unittest discover tests
       displayName: 'Install dependencies and run tests'
   ```

2. Commit and push the pipeline file. Azure DevOps will automatically detect it and trigger a build on push events.

## GitHub Copilot Support

If you develop in VS Code or a JetBrains IDE, you can leverage **GitHub Copilot** even when your code is hosted in Azure Repos. Simply clone the repository locally, open it in your IDE, and ensure that the Copilot extension is installed. Copilot will work with the local files and provide coding suggestions as you implement new features.

## Tips for Teams

- Configure branch policies to protect the main branch from direct commits.
- Use Azure Boards dashboards to visualize progress and track ongoing tasks.
- Link MLflow experiments to Azure DevOps work items by referencing experiment IDs in task descriptions.

This integration will help ensure visibility, collaboration, and accountability as you develop the MLOps solution accelerator.
