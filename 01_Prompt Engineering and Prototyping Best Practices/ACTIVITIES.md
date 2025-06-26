<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

<h1 align="center" id="heading">Session 1: Introduction and Vibe Check</h1>


##### 🏗️ Activity #1:

Please evaluate your system on the following questions:

**NOTE**
    - System prompt given: **You are a helpful assistant.**
    - User promts: **See below**

1. Explain the concept of object-oriented programming in simple terms to a complete beginner. 
    - Aspect Tested: 
        * **This prompt is testing if the application can handle explanation of open and objective knowledge efficiently to the level requested. In this case that level would be a complete beginner.**
        * **The type of expected answer allows room for testing display style. In this case the answer includes a markdown style that is not shown properly. It is displayed as plain text. This needs to be fixed.**

2. Read the following paragraph and provide a concise summary of the key points…
    - Aspect Tested:
        * **If we replace the ... in the prompt by a given paragraph we would be testing summarization capabilities of the app as well as display style for a summarization use case. In our specific case the summarization was almost as long as the original paragraph. This may not need to be fixed at the app level, but maybe more as we modify the system prompt."**
        * **This prompt if taken literally, it contains ambiguous or missing information aiming to check how the system would react to that situation (corner case): would it give a reasonable answer asking for what is missing? In this case it does that: "Of course! Please provide the paragraph you'd like me to summarize."**

3. Write a short, imaginative story (100–150 words) about a robot finding friendship in an unexpected place.
    - Aspect Tested:
        * **This prompt tests how well the app follows our specific requests: (a) short 100-150 words story, (b) about a robot finding friendship in an unexpected place, (c) imaginative. Results from the prompt hit all of the requirements at first glance.**. 
        * **Imagination\Originality: Digging a little deeper, when asking the same question again, the app provides a very similar story from the point of view that a robot and a cat meet at a scrapyard. The common elements highlight that this may not be a very imaginative store considering the request. We can add a parameter for adjusting model temperature via a "creative" slider on the front end (0-10).**. 

4. If a store sells apples in packs of 4 and oranges in packs of 3, how many packs of each do I need to buy to get exactly 12 apples and 9 oranges?
    - Aspect Tested:
        * **This prompt tests mathematical reasoning capabilities of our app (model). This includes not just reasoning, but also objective correctness**. 
        * **It also test display style for an answer to mathematical problem**. 

5. Rewrite the following paragraph in a professional, formal tone…
    - Aspect Tested:
        * **If we replace the ... in the prompt by a given paragraph we would be testing: (a) rewriting capabilities of the app, (b) how well does the app follow specific requests such as "professional and formal tone", and (c) display style for a rewriting use case."**
        * **This prompt if taken literally, it contains ambiguous or missing information aiming to check how the system would react to that situation (corner case): would it give a reasonable answer asking for what is missing? In this case it does that: "Certainly! Please provide the paragraph you would like me to rewrite in a professional, formal tone."**


##### 🚧 Advanced Build:

Please make adjustments to your application that you believe will improve the vibe check done above, then deploy the changes to your Vercel domain 

* **Done. App can be found at: https://ai-engineering-challenge-self.vercel.app/**

##### 🧑‍🤝‍🧑❓ Discussion Question #1:

What are some limitations of vibe checking as an evaluation tool?

* **It does not offer full test\check coverage**.
* **It is a manual process, meaning time consuming**. 
* **It does not produce objective numerical scores for sharing with others - we can only share vibes**. 
* **It can be subjective depending on who is doing the checking\testing. Different testers can have\experience different vibes**. 
