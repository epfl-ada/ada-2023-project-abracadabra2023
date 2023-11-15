# A world full of clichés

## Abstract:
Clichés, in this context, refer to common ideas and concepts that quickly come to mind when thinking about a particular subject. In Wikispeedia, these clichés may manifest either as individual pages or as page categories typically associated with a shared concept and that players know should contain relevant hyperlinks for their target.
The core concept of our project involves identifying sets of potential clichés related to various topics, such as countries, and assessing their frequency of occurrence in actual games. We aim to determine how frequently players rely on these clichés and whether they contribute to players successfully reaching their targets.

## Research questions:
- How can we select the clichés of a given topic without being too much biased?
- Do clichés influence our searching process when we know our goal?
- Can clichés help us determine if some articles are badly written, that is if they need more hyperlink?

## Additional datasets

## Methods:
Before starting our initial analysis, we hate to clean pre-process our dataset, in order to obtain significant and meaningful results. To do so, we create a framework that permit us to analyze the different paths and articles. We can carry out an analysis at different levels (unit can be page, category, subcategory, sub-subcategory, etc.) that can consist in:
- Choose a main unit, i.e., an article. The goal would be to compare the analysis for different main units to compare the results, but in this step of the project we would only look at one to verify that it is feasible. Here we chose to study the main unit ‘United_Kingdom’.
- Define a set of units consisting of clichés associated with the main unit. This can be achieved manually, with a 100% manual selection from all the in and out neighbors of the primary unit. Alternatively, empirical subsets of pages can be employed to identify these clichés (e.g., selecting the most common ones in the paths taken). However, solely relying on clichés from the provided dataset may introduce bias, as it would only capture player-specific clichés rather than more general ones. Therefore, an alternative approach involves examining clichés beyond the dataset and associating them with an article or a category to broaden the scope of our analysis. Another option is to randomly generate clichés for a particular unit, for instance by using ChatGPT.
- Defining performance metrics based on actual success of the path, difficulty rating, length of the taken path, length of the actual shortest path, number of backtracks, etc.
- Comparing the performance of games passing by the main unit and analyzing the correlation with the usage of the cliché’s units. By defining thresholds, we could compute binary classification metrics like precision, recall, etc.
- (inverse problem) Find other sets of units that increase or decrease the precision / recall metrics and see whether these units are intuitively clichés or not.

## Proposed timeline:
- 17.11.2023: Project milestone 2 deadline
    - Data pre-processing and cleaning have been completed and initial data analysis started. 
- 01.12.2023: Deadline Homework 2
    - Pause of the project to work on the homework until the homework deadline.
- 08.12.2023: Finish the statistical analysis and start the visualizations. Start to create a draft for the data story.
- 15.12.2023: Work on the data story and the website.
- 19.12.2023: Finish the data story, complete the final notebook and other useful documents. Update the README.
- 22.12.2023: Project milestone 3 deadline

## Organization within the team:
- Tudor:
- Anna:
- Martin:
- Philippe:
- Salya:

## Questions for TAs:
- Is ChatGPT enough to obtain clichés without being too subjective and biased?
