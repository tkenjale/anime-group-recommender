We can consider developing the interactive visualization frontend as developing a simple website with specific features/capability. So, the task can be considered as developing a product - a recommendation website, thus justifying the existence of this document. This file can also serve as content for the poster/presentation.

# 1. Context

Requirements can be looked at from the perspective of the end users; this can often be described in the format: As a ... I want ... So that ...

**As a** group of friend sharing interest in anime

**We want** to have a recommendation tool to find some anime series that fit our interest

**So that** the whole group can watch together

Taking that need from the end users, a product owner with flesh out the product requirements in the following ways: 
- Feature backlogs: Features are functional capabilities that the product need to have
- NFRs (Non-functional requirements): How to make sure the developed features correlate with user experience

# 2. Features

## 2.1. Front-end
For the group recommendation website front end, it needs the following features (this is already extremely reductive):
- An interactive UI, on which the users can:
  - Add a new group member
  - Enter the name of the new group members
  - Add anime and associated ratings for the new group member, including:
    - Movie name: Text input
    - Rating: Float input, range between 1 - 10
  - Add anime and associated ratings for an existing group member (similar above)
  - Remove anime and associated ratings for an existing group member
  - Change the rating for a movie of an exisitng group member
  - Change the name of an existing group member
  - Choose options for recommender algorithms:
    - Regularization factors
    - Aggregation step: Before factorization (BF) or After fatorization (AF)
    - Aggregation function: Average, Most happiness, Least misery
  - A **Submit button** to send the input to the backend
- A visualization of the recommendation, displayed after the user click "Submit"
  - Description: The visualization will include the following panels:
    - Member panels: Each contain the member name and a list of anime + ratings of that member
    - A recommendation panel: Lies in the middle of the page, containing the top 10 recommendation and associated recommendation strengths
    - Links between each member panel and the recommendation panel
  - The list of anime needs to include the following details (left to right)
    - An image representing that anime
    - Correct name of that anime in our database
    - Rating displayed as a float
  - When hover over the member-recommendation link, display the predicted rating of that member for those 10 anime

## 2.2. Back-end / API
To be able to adapt on the front end, we need to work out the following features in the backend:

- ***To be added***

## 2.3. Algorithms
To be able to adapt on the front end, we need to work out the following function for the algorithms:

- Before Factorization method:
  - `[Done]` Load data, preprocess and solve ridge regression for the virtual user
  - `[Done]` Post process, sort and return top 10 for the virtual user
  - `[Todo]` Add option to choose social functions to aggregation method (currently only average)
  - `[Todo]` Add recommendation for individual users
- After factorization method:
  - `[Todo]` All features similar to BF

# 3. Non-functional requirements

- Latency: Less than 1s from user click "Submit" to visualization display
- ***To be added***

