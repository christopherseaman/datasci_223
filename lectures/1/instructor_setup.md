# GitHub Classroom Setup Guide for Instructors

This guide explains how to set up the GitHub and Python Foundations assignment using GitHub Classroom.

## Prerequisites

1. A GitHub account with GitHub Classroom access
2. A GitHub organization for your course (can be created through GitHub Classroom)

## Setting Up the Assignment

### Step 1: Create a Template Repository

1. Create a new repository in your GitHub organization
2. Name it something like "github-python-foundations-template"
3. Upload the contents of the `template` directory to this repository:
   - README.md
   - email_hasher.py
   - tests/ directory
   - .github/ directory
4. Make sure the repository is public or accessible to your organization members

### Step 2: Create a GitHub Classroom Assignment

1. Go to [GitHub Classroom](https://classroom.github.com/)
2. Select your classroom or create a new one
3. Click "Create assignment"
4. Fill in the assignment details:
   - Assignment title: "GitHub and Python Foundations"
   - Assignment type: Individual
   - Repository visibility: Private
   - Select the template repository you created in Step 1
   - Enable feedback pull requests
   - Enable auto-grading (it will detect the tests automatically)

### Step 3: Distribute the Assignment

1. After creating the assignment, GitHub Classroom will generate an invitation link
2. Share this link with your students through your course management system
3. When students click the link, they will:
   - Be prompted to accept the assignment
   - Get a private repository with the template code
   - Be able to work on the assignment and push changes

## Monitoring Student Progress

### Viewing Auto-grading Results

1. Go to your GitHub Classroom dashboard
2. Select the assignment
3. You'll see a list of students and their submission status
4. Click on a student's repository to view their code
5. Go to the "Actions" tab to see the auto-grading workflow results
6. Each test will show as passed or failed, with detailed logs available

### Providing Feedback

1. You can provide feedback by creating issues on students' repositories
2. You can also use the feedback pull request feature to comment on specific lines of code
3. For more detailed feedback, you can clone their repository locally and run the tests yourself

## Troubleshooting

### Common Issues

1. **Tests not running**: Make sure the workflow file (.github/workflows/autograding.yml) is in the correct location and properly formatted
2. **Students can't access the assignment**: Check that they're members of your GitHub organization or that the assignment is set to public
3. **Tests failing unexpectedly**: Check the test logs for details and update the tests if necessary

### Support Resources

- [GitHub Classroom documentation](https://docs.github.com/en/education/manage-coursework-with-github-classroom)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [GitHub Education support](https://education.github.com/teachers)
