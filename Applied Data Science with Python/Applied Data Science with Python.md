# Course links

- [[Syllabus- DataSci 223]]
- GitHub - [https://github.com/christopherseaman/datasci_223](https://github.com/christopherseaman/datasci_223)
- Discord - [https://discord.gg/QdTYgR45er](https://discord.gg/QdTYgR45er)
- Project - [[Project]]
- Submit assignments - [https://forms.gle/EmcM9s7UrVGLykDm9](https://forms.gle/EmcM9s7UrVGLykDm9)
- Recordings - [https://lecture.ucsf.edu/](https://lecture.ucsf.edu/ets/Channel/b28d58f224e040c59832d99d7d91aa995f/browse/null/most-recent/null/0/null)
- This page - [https://not.badmath.org/ds223](https://not.badmath.org/ds223)

# Reference materials

## Recommended

Python: general introduction

- _Whirlwind Tour of Python_, VanderPlas - author’s [website](https://jakevdp.github.io/WhirlwindTourOfPython/)
- _Think Python_, Downey - purchase or read at [Green Tea Press](https://greenteapress.com/wp/think-python/)
- _Hitchhiker’s Guide to Python!_ - official [documentation](https://docs.python-guide.org/)

Python: data science and analysis

- _Python for Data Analysis_, McKinney - author’s [website](https://wesmckinney.com/book/)
- _Python Data Science Handbook,_ VanderPlas - author’s [website](https://jakevdp.github.io/PythonDataScienceHandbook/)
- _PyTorch Tutorials_ - official [documentation](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- _TensorFlow Tutorials_ - official [documentation](https://www.tensorflow.org/tutorials)
- _Dive into Deep Learning -_ authors’ [website](https://d2l.ai)
- _Understanding Deep Learning_ - author’s [website](https://udlbook.github.io/udlbook/) (**WARNING:** intense math)

[O’Reilly Library Access](https://www.oreilly.com/library-access/) (UCSF Institutional login)

- [Hands-on Machine Learning, Géron](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) and companion [repository](https://github.com/ageron/handson-ml3)
- [Machine Learning with PyTorch and Scikit-Learn, Rashka](https://learning.oreilly.com/library/view/machine-learning-with/9781801819312/)
- [Deep Learning with PyTorch, Viehmann](https://learning.oreilly.com/library/view/deep-learning-with/9781617295263/)

## UCSF Library

[UCSF library](https://www.library.ucsf.edu/) resources that might be helpful for your class project or future work:

- **Library Databases Catalog**: [Library databases by subject/type/provider](https://guides.ucsf.edu/az.php)
- **Research Guides**: [Great for finding data sources for population health, health service research, and COVID-19 research](https://guides.ucsf.edu/)
- **UC Data Week Seminars**:
    - [Digital Humanities Data](https://docs.google.com/presentation/d/1vYqIOm0YLVvosmKTDMmTeEDijmbkRE6OIqD0bishPo0/edit#slide=id.gbc0c25baf6_0_1997)
    - [Social Sciences/Government Data](https://docs.google.com/presentation/d/17uks6NcY-5kJWYKl0sd5pITd4caZp4vSes7HjwFiCWk/edit#slide=id.p1)
    - [GIS Data and Mapping](https://docs.google.com/presentation/d/1doAGDZyUVFrHGkGUf9o0A13UflLGYt-7s-DLQpVzwz0/edit#slide=id.p)
    - [Web Archives as Data](https://docs.google.com/presentation/d/1VW6vpZWEiKu7O55pWAAbafXYvrJ82_0EDsg-pnsWrE8/edit#slide=id.p)
    - Recordings and more materials available for [search](https://uc-love-data-week.github.io/search)

If you are working with EHR data from the info commons, in addition to the

[wiki](https://wiki.library.ucsf.edu/display/IC/UCSF+Information+Commons+Wiki), this page might be worth checking out:

- **Informatics Tools**: [Seminar materials](https://courses.ucsf.edu/course/view.php?id=7912#section-6), including access to high-performance computing, research analysis environment, info commons, patient explorerR, imaging commons, EMERSE for querying and analyzing clinical notes, etc.
- **Office Hours related to data assets and info commons**: [schedule](https://wiki.library.ucsf.edu/display/IC/Support#Support-OfficeHours)

The data science and open scholarship team also has help desk hours and consultations for related questions: [DSOS team](https://www.library.ucsf.edu/ask-an-expert/data-science/)

## Additional options

Python: general introduction

- _Python 101_ - community resource on the [web](https://python101.pythonlibrary.org/)
- _Python for Everybody_ - author’s [website](https://www.py4e.com/)
- [pythonbooks.org](https://pythonbooks.org/free-books/) - additional free books available
- _Learning Python_ - great (but not free) book [from O’Reilly](https://www.oreilly.com/library/view/learning-python-4th/9780596805395/)
- _Python Cookbook_ - applications and examples (not free) [from O’Reilly](https://www.oreilly.com/library/view/python-cookbook/0596001673/)

Python: data science

- _Machine Learning from Scratch_ - author’s [website](https://dafriedman97.github.io/mlbook/content/introduction.html), learn the nuts and bolts behind the tools
- _Artificial Intelligence: A Modern Approach_ - author’s [website](https://aima.cs.berkeley.edu/), authoritative deep survey of AI topics

Python: library-specific

- _Effective Pandas_ - author’s [website](https://leanpub.com/effective-pandas), deep dive into a common python library not covered in this course
- _Deep Learning with PyTorch_ - read on publisher’s [website](https://www.manning.com/books/deep-learning-with-pytorch)

# Topics

1. [[Getting started with git, python, and R]]
    
    - In brief
        
        - Tools: `python`, `R`, and `git`
            - Getting set up locally
            - Cloud options (Colab, Binder, Paperspace, GitHub)
        - Markdown
            - Syntax summaries
            - GitHub and Notion flavors
            - Readme.md - make one for every repo
        - git and GitHub
            - Starting or cloning a repository
            - Git push/pull/sync
            - Branches
            - Conflicts
        - Python
            - Syntax basics
            - Running python and jupyter
            - Variables and control flow
            - Common packages
        
    
    - [**Lecture 1 Overview**](https://www.notion.so/Lecture-1-Overview-44c1d71be20f4937a84dbe031c15cbfa?pvs=21)
2. [[Cleaning, munging, and wrangling]]
    
    - Agenda
        
        - X-ray vision
        - Packages - the best code (you didn’t write)
        - Care and feeding of pandas
        - Cleaning, munging, and wrangling
        - Putting labels on things
        - Getting dirty with MNIST
        
    
    - [Lecture 2 Overview](https://www.notion.so/Lecture-2-2b9f1154960e404e80cefed31e6ef9f3?pvs=21)
3. [[JOIN the DISTINCT with SQL]]
    - [Lecture 3 Overview](https://www.notion.so/Lecture-3-497daac8c651447fa876bd54562bc1d9?pvs=21)
4. [[Putting a label on things]]
    
    - Agenda
        - Quick example
        - Classification model types and how to evaluate them
        - How things can go wrong…
        - … and how to fix it
        - Hands-on with code
    
    - [Lecture 4 Overview](https://www.notion.so/Lecture-4-a8840e89eba2414593b05bcb7fe5fc41?pvs=21)
5. [[Classification, continued]]
    - [Lecture 5 Overview](https://www.notion.so/Lecture-5-Lecture-4-cont-d-6b685fdad4df4ec29f30c07f89597daf?pvs=21)
6. [[Neural Networks- If I only had a brain]]
    - [Lecture 6 Overview](https://www.notion.so/Lecture-6-0a23c37715414365bf7d3c1ffa885aa0?pvs=21)
    - [[Keras Cheat Sheet]]
    - [[PyTorch Cheat Sheet]]
7. 🤖 **[Intro to Deep Learning Transformers](https://docs.google.com/presentation/d/1hhrsbrtRS-S2LXBEOXzKJ0Azg9IOetNLKTejIyV2FyE/edit?usp=sharing)** with Tony Doran
    - [Lecture 7 Overview](https://www.notion.so/Lecture-7-6fd8a4cf069d47fda841c48df846b79f?pvs=21)
8. [[Jobs & Impostor Syndrome]]
9. **FINAL LECTURE!**
    
    1. [[Somewhere over the sea…]]
    2. [[Embeddings]] with Orpheus Mall
    3. [[Deploying models]]
    
    [[It’s About Time]]
    

## Potential additional topics

1. Embeddings deep dive
2. Experimentation & A/B Testing
3. Time Series Analysis and Forecasting
4. Generative AI with Images
5. Data Visualization and Communication
    
    1. [Visualization](https://colab.research.google.com/github/khuyentran1401/Efficient_Python_tricks_and_tools_for_data_scientists/blob/master/Chapter5/visualization.ipynb) (tour)
    
    - [Visualization](https://colab.research.google.com/drive/1jky2SKKPp2AWvldoKPggLT0DLSVfG3HQ) (matplotlib)
6. Feature Engineering and Selection
7. [[Algorithms]]
8. Natural Language Processing (NLP)
9. Local setup with the "Modern Data Stack"
10. Deploying a basics model/app to the web
11. Computer vision basics (image classification → tiny real-world demo)
12. Big Data and Distributed Computing
13. Ethics in Data Science

## Archive

Topics from previous incarnations of the course

- [[Guest Lecture- Anthony Doran of SimplyPut]]

[[Project]]