# A world full of clichés

_abracADAbra2023: Tudor Oancea, Anna Schmitt, Martin Leclercq, Philippe Servant, Salya Diallo_

## Abstract:
Clichés, in this context, refer to common ideas and concepts that quickly come to mind when thinking about a particular subject. In Wikispeedia, these clichés may manifest either as individual pages or as page categories typically associated with a shared concept and that players know should contain relevant hyperlinks for their target.
The core concept of our project involves identifying sets of potential clichés related to various topics, such as countries, and assessing their frequency of occurrence in actual games. We aim to determine how frequently players rely on these clichés and whether they contribute to players successfully reaching their targets.


## Research questions:
- How can we select the clichés of a given topic without being too much biased?
- Do clichés influence our searching process when we know our goal?
- Can clichés help us determine if some articles are badly written, that is if they need more hyperlink?


## Methods:
Before starting our initial analysis, we hate to clean pre-process our dataset, in order to obtain significant and meaningful results. To do so, we create a framework that permit us to analyze the different paths and articles. We can carry out an analysis at different levels based on a main *unit*, that can be a page, category, subcategory, sub-subcategory, etc. Our analysis will can consist in:
- Choosing a main unit, which will just be an article in this milestone. The goal would be to compare the analysis for different main units to compare the results, but for now we only look at one to verify that it is feasible. We chose to study the main unit ‘United_Kingdom’.
- Define a set of units consisting of clichés associated with the main unit. This can be achieved manually, with a 100% manual selection from all the in and out neighbors of the primary unit. Alternatively, empirical subsets of units can be employed to identify these clichés (e.g., selecting the most common ones in the paths taken). However, solely relying on clichés from the provided dataset may introduce bias, as it would only capture player-specific clichés rather than more general ones. Therefore, an alternative approach involves examining clichés beyond the dataset and associating them with an article or a category to broaden the scope of our analysis. Another option is to randomly generate clichés for a particular unit, for instance by using ChatGPT.
- Defining performance metrics based on actual success of the path, difficulty rating, length of the taken path, length of the actual shortest path, number of backtracks, etc.
- Comparing the performance of games passing by the main unit and analyzing the correlation with the usage of the cliché’s units. By defining thresholds, we could compute binary classification metrics like precision, recall, etc.
- (inverse problem) Find other sets of units that increase or decrease the precision / recall metrics and see whether these units are intuitively clichés or not.
- A final step adressing the last one of our research questions would be to check if among the units (in particular pages) that we considered as clichés, there is an increased proportion of backtracks performed by the players (outlier detection based for example on one-class SVM). This would indicate that the players chose this page because they thought it will contain a certain hyperlink but it did not. We could then inspect the actual page content to look for words corresponding to clichés that should have been hyperlinked. Other analysis based on the future lectures could be performed to detect if the page is badly written (e.g. if it contains too many clichés, or if it is too long, etc.).

## Additional datasets:
As stated above, we need to select some clichés independently from the dataset to compare them from the ones we will find using the Wikispeedia dataset. To do so, we will consider the following strategies:
- We can use the ChatGPT model to generate clichés using different prompts. We can then select the most relevant ones.
- Browse the internet to find articles about clichés on our topic, for instance, there is exists a Wikipedia page about [Stereotypes of british people](https://en.wikipedia.org/wiki/Stereotypes_of_British_people). And then select the most relevant ones.
- Finally, we find an external dataset called [SeeGULL](https://github.com/google-research-datasets/seegull) which measures the offensiveness of clichés on geographical identity groups. The stereotypes contained in the dataset were generated using PaLM and GPT-3. The interesting part for us is that it contains an identity group named 'British' with some possible stereotypes associated to it, and each of these stereotypes are humanly evaluated as stereotypical, non-stereotypical or unsure. Finally the annotators are divided between European and North-American people, this provides two point of view, one from the inside and one from the outside. Hence, We can use this dataset to select the most relevant clichés, while keeping in mind that it refers on British people and not on the United Kingdom.


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
- Tudor: website, ML, code quality
- Anna: sublime plots, data story
- Martin: statistical analysis and ML, code quality
- Philippe: website, ML
- Salya: data cleaning and pre-processing, data story


## Questions for TAs:
- Is ChatGPT enough to obtain clichés without being too subjective and biased?
