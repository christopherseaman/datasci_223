## For Exercise 4 - Krish Rai and Taimoor Qureshi

### Task 2

Sorry if it's a bit messy! Was jumping between two files because the fitting took long to run so some labels or lines may be unclear.

This is here: run from top and stop where the title says Task 3. That is a different file.

https://github.com/kiar24/datasci_223/blob/main/exercises/4-classification/exercise4_5Task2.ipynb

1.Run everything from top and make sure all packages (including classification) are installed or loaded. Uncomment if a package is hidden

2.This was done in Google Colab, so this may need to be done everytime it disconnects. Otherwise, download and open directly on computer.

3. Load provided code. 

4. For Task2, model training was difficult because even the subset data (68795 rows of a-g lowercase data) took forever to run through grid search.

5. In order to speed it up it was run on 6,000 and 20,000 rows. The code is set to 20,000 currently. If you want to reduce rows, change a2gagain = a2g.iloc[:20000,:] to another number instead of 20,000. The smaller it is the faster it will run, but it will be less data.

6. Went with random forest because it tends to do well and the parameters are easy to adjust. Logistic regression works better with binary and boost and other methods were not used as forest did well.

7.Random_state was set when splitting data. Change this if you want to play around.' Grid search was used.

8. Accuracy for this model was 0.949. 

9. On test it did very well, virtually all correct (0.99). This may also be due to the random_state split. But the testing did well per confusion matrix and accuracy.

10. Since there were few errors, there was not much to improve. Also, since it is subset there were no clear letters that were that similar (maybe b and d but this did not get misclassified).

11. This was rerun on entire 68795 rows with updated parameters to reevaluate. Accuracy was 0.961 so it improved. This took over an hour to run.

### Task 3
This is here: run code up until Problem 1 Part 1: Subset, Cleaning, and Choosing a Model title (this is Task 2). Start running the code when the Task 3 header shows up. You can run Task 2 again if you want, but it just takes time because of the rows.

https://github.com/kiar24/datasci_223/blob/main/exercises/4-classification/exercise4_5assignmentTask3.ipynb

1. Tried using logistic regression because of binary classification, but it failed to converge. When I tried scaling, there were many errors.

2. So Random forest was chosen as described in Task 2. This was done two ways, binary (upper vs lower case prediction) and character prediction (similar to Task 2).

3. Grid search used again. Best parameter chosen score was 0.82. Evaluated on validation set to 0.81. Binary confusion matrix shown.

4. This was then repeated to predict each character the same as as Task 2. Accuracy 0.794 was best score when fitting, and 0.78 on validation set evaluation.
