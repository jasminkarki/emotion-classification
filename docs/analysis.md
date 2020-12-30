## Project Analysis
### User story structure:
> => [Who? -> What? -> Why?] 

* As a teacher, I want to maximize the number of happy students, so that I can increase the number of attendees in my class.
* As a policymaker, I want to classify student’s reviews, so that I can make policies to increase the reputation and rank of the college.

### Problem and Solution Formulation
**What is the problem?**
*Informal description:*
* I need a program that will tell me the emotion of the status posted.

*Formalism:* 
* Task(T): Classify students posts to detect emotions
* Experience(E): A corpus of posts in the chatroom
* Performance(P): Precision, recall

*Assumptions:*
* If the post contains positive words like: 
    > excited, sensuous, energetic, enthusiastic, etc. the emotion is happy. 
    > guilty, ashamed, depressed, etc. the emotion is sad.
    > hurt, hateful, selfish, hateful, etc. the emotion is angry and so on


**Why does the problem need to be solved?**
*Motivation:*
* Many students are unsatisfied with the curriculum they are studying, the teaching method, and the policies and rules created by the HOD or management of the college, so there is a higher chance of low attendance, poor performance, and even increased dropout rates affecting the ranking and reputation of the college. 

*Solution Benefit:*
* It will increase class attendance and thus, the performance of students.

*Solution Use:*
* The solution will be used to increase the ranking and reputation of the college. The lifetime of the program will be around two years for now. 


*How would I solve the problem?*
* Explore how you would solve the problem manually.*
* We read the status of the students. 
* We identify the source of satisfaction/dissatisfaction (curriculum, teacher, department, management) 
* We identify the emotion and tone based on the keywords used in the status. 



*Explore how you would solve the problem by using machine learning tools.*
* Task: Emotion classification system
* Approaches:
    > i. Investigation and building of machine learning model.
    > ii.  Investigation and building of deep learning model.
