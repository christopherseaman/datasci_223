# A Guide to Debugging Python code (and why you should learn it)

Any time you start writing code for your application, everything works from the first try, you never encounter exceptions and errors, users are happy and so are you. This would be wonderful, wouldn’t it? Unfortunately it’s only possible in a dream. The reality is that your code fails. It raises exceptions, it produces unexpected outcomes, it makes users unhappy.

This is why debugging skill is essential in a developer’s tool kit. Debugging is a process of determining and fixing errors in your code. And while basic errors can be spotted by eye in simpler code, as the application is getting more complicated, we need tools to help. This article discusses the importance of these tools and why you need to master them right now.

# Who the f… is the debugging?

Application issues generally fall into two categories: raising exceptions and silently doing things wrong. Sometimes, exceptions are clear, revealing mistakes immediately. Yet, for complex cases the answer is not on the surface, be it a weird exception thrown or an absolutely unexpected result of a program run encountered. And this is where we need to debug.

There are several options for fixing your code. The “easiest” way is to chaotically change different parts of it, hoping for the best. Sometimes it helps, but to be honest, this one is too inefficient and will probably waste a ton of your time.

To efficiently make changes, understanding the problem and, hence, the possible source of it, is the key. We begin this with studying the traceback. Here, when you have understood the problem you might start staring at your code, trying to think the solution out. Maybe you will add print()’s here and there, and maybe they will even give you a clue. This approach is legit when your application is simple, but, surprisingly, many developers stick to it even after years of working, spending more and more time with the code getting more complex.

The third option is on the ‘evolutionary’ top of the previous two: the proper debugging as a union of tools, approaches and experience.

# The goal of debugging and types of tools

The ultimate goal of debugging is to discover the issue and figure out how each line of code affects the corrupted flow, how each value changes as the program keeps running. After this the only thing that’s left is implement a better solution that covers the edge cases.

Debugging involves setting breakpoints, stepping through code and inspecting values. Existing debugging tools can be split into three main types:

- standalone UI applications — for example, PuDB, Winpdb. These are applications designed specifically for debugging, and the UI is basically the same as in IDE, which then, in my opinion, is not worth wasting your time. However, you can always learn more about it by yourself if you’re interested;
- code-in tools, where you set breakpoints by adding temporary code to your program;
- IDEs. All of the popular modern development environments, such as PyCharm, Visual Studio, PyDev and many more, have integrated debugging tools, allowing you to run and inspect the code at the same time.

Now let’s talk a bit more about the last two points.

# An example

To explain how to debug in action, I’ll show you the process with the example code below.

Let’s say you’re creating an application to work with some external weather API. You build a pydantic model to load the API response in it to use the data afterwards (for example, print it to the console). Take a look at the example. This code doesn’t work.

from pydantic import BaseModel  
  
class ApiResponse(BaseModel):  
    city_id: int  
    temperature: float  
    weather: str  
  
# we don’t see the contents of this method, pretend this is what an actual HTTP API call returns  
def fake_weather_api_request(city: str):  
    cities = {"New York": {"id": 42, "temperature": 25.6, "weather_type": "sunny"},  
              "Marrakech": {"id": 12, "temperature": 30}}  
    return cities.get(city)  
  
if __name__ == "__main__":  
    weather_resp = fake_weather_api_request("New York")  
    ny_weather = ApiResponse(**weather_resp)  
    print(f"Weather in New York (city id={ny_weather.city_id}): \n" +  
          f"Temperature: {ny_weather.temperature} \n" +  
          f"Weather type: {ny_weather.weather}")

When you run it, you will see the following exception message:

![](https://miro.medium.com/v2/resize:fit:1400/1*Jq6t21z0GVGCOZb5cRCJeQ.png)

Oh. What would that mean? If we’re going the easy way and trying to guess the problem, we might think — probably, the following fields don’t come with a response for certain cities. Let’s make them nullable!

class ApiResponse(BaseModel):  
    city_id: int = None  
    temperature: float = None  
    weather: str = None

Awesome. It’s not failing anymore.

![](https://miro.medium.com/v2/resize:fit:1040/1*ie6BpdoNqU2tZEJwUoFPPg.png)

Or is it? What if we know for sure the API has both city id and weather type info for New York? Why is it None then? We could continue playing a guessing game, but this is actually the perfect moment to start debugging.

# The Python Debugger

Python provides you with an awesome built-in code debugger: the **pdb**module. This is an easy to use interactive Python code debugger. In Python versions prior to 3.6, we would need to import the module in the beginning of the file to use the debugger. In modern Python, though, even that is not necessary anymore — simply putting the built-in `breakpoint()` function call whenever you want your program to pause will work.

Let’s set a breakpoint before the API call.

if __name__ == "__main__":  
    breakpoint()  
    weather_resp = fake_weather_api_request("New York")  
    ny_weather = ApiResponse(**weather_resp)

Now you can run the application from the terminal:

![](https://miro.medium.com/v2/resize:fit:1400/1*MUefNGueMajr6nMEj2lgMA.png)

The application execution has paused at the place where you set the breakpoint, and the pdb is now awaiting for your command input. Type **“h”** to view the list of available commands:

![](https://miro.medium.com/v2/resize:fit:1400/1*YESV-Eb1nEeuESKUrSqmDQ.png)

Now you can navigate over your code, print variables values at different stages of program execution, step into methods to see what’s happening inside, etc.

Let’s step over the API call by using the command **n (next)** and then see what was the actual response body with **p**.

![](https://miro.medium.com/v2/resize:fit:1400/1*XjfpTZMw3x22euKNkFSv_A.png)

Okay, it turns out, the response does actually contain all three fields, but why are some of them displayed as `None` when printing it out? If you look closely at the `ApiResponse` model defined in the beginning of the file, you will see that some of the fields names are slightly incorrect: **city_id** instead of **id**, and **weather** instead of **weather_type**. And the actual API response indeed doesn’t contain these fields. Let’s fix this.

class ApiResponse(BaseModel):  
    id: int = None  
    temperature: float = None  
    weather_type: str = None

And don’t forget to change field names in the print statement and remove/comment the `breakpoint()` call:

if __name__ == "__main__":  
    #breakpoint()  
    weather_resp = fake_weather_api_request("New York")  
    ny_weather = ApiResponse(**weather_resp)  
    print(f"Weather in New York (city id={ny_weather.id}): \n" +  
          f"Temperature: {ny_weather.temperature} \n" +  
          f"Weather type: {ny_weather.weather_type}")

![](https://miro.medium.com/v2/resize:fit:1400/1*udSZ9NEuWkIHBR_Q6AkdWA.png)

After you run the code again — yay, it works!

Congratulations, you’ve just successfully finished your first round of debugging. :)

# Debugging in IDEs

An IDE (integrated development environment) helps you develop software efficiently. It combines all of the main functions needed for software development — code editing, building, running, testing and, finally, debugging. IDEs make debugging considerably easier so unless you’re a rigid fan of a terminal, I would recommend trying debugging in an IDE.

Most IDEs operate on similar principles and it doesn’t make sense to review all of them. So I will show you a few examples in PyCharm, my preferred IDE, but you can easily do the same in any other development environment of your choice.

Let’s open the same program code in PyCharm. To set a breakpoint, you need to click on the left of the line you want to pause at.

![](https://miro.medium.com/v2/resize:fit:1400/1*12cFrm9v8mlSHBwr2jJxUg.png)

This will stop the program executing after receiving the API response. To see it in action, hit the green bug button in the upper right corner of the window.

![](https://miro.medium.com/v2/resize:fit:1400/1*GnrSUQhwKgO0HZD0G0oHeA.png)

When the application execution flow will reach the marked line of code, PyCharm will open a debugging sub-window containing useful things.

![](https://miro.medium.com/v2/resize:fit:1400/1*lEW9zKZbrmT-GFZ6twjabw.png)

On the picture above, (1) is the list of currently defined variables and their values. Each variable can be expanded and deeply examined on all levels. This is extremely useful when the variable value is a complex object with nested values of different classes. (2) is the latest received/changed value. This is helpful for fast checking, if the value is simple as a dictionary like in our case or a string. And last, but not the least, at (3) you can see the toolbox for stepping through the code. Their description together with some additional documentation on debugging in PyCharm can be found on the official website: [https://www.jetbrains.com/pycharm/features/debugger.html](https://www.jetbrains.com/pycharm/features/debugger.html).

I, however, would like to pay extra attention to the tiny calculator button next to all the stepping buttons. This button is called “Evaluate expression” and the name is self-explanatory. This will save you a lot of time when you want to see different outcomes of an expression without changing and re-running your code. In our weather API example we could use the tool to see what might be the response for a city other than New York.

![](https://miro.medium.com/v2/resize:fit:1400/1*KJ7Z5D_FDTO_qVihyKyNYA.png)

Voila, we can see that for some cities, for example, Marrakech, the weather_type attribute is missing.

We could also try loading the response into the ApiResponse model right away and see how it would behave.

![](https://miro.medium.com/v2/resize:fit:1400/1*zIqMwmI9siZTubJ_W9iARQ.png)

Turns out, for Marrakech the `weather_type` field will be None.

If you try to evaluate an incorrect expression, PyCharm will show you an error. But this error is only visible within the “Evaluate” window and won’t break the program flow. After you’ve checked everything you wanted, you can close the window and continue stepping through the code just like before trying out those things.

![](https://miro.medium.com/v2/resize:fit:1400/1*2I1WesQZ8cZpzmtR2cuWTg.png)

The debugging skill is only gained through experience, but once you embrace it, you’ll be surprised you managed without it for so long. Making mistakes is absolutely natural, and writing and writing bug-free code is nearly impossible (if you code more than a couple of lines a year). What truly matters is your ability to investigate the problem and fix the issue.

Did you find this post helpful? Hit clap and follow to read more of them later :)