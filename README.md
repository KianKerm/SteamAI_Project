# [Steam Personal AI Assistant]
Try the app here! [https://steamai.streamlit.app/]
An OpenAI-powered assistant that helps you explore your Steam library, understand your gaming habits, and find something to play.
Built with Python, Streamlit, pandas, and the Steam Web API.
Explore Your Steam Library
Enter a Steam profile to get started. The app brings your library, playtime insights, and AI chat into one interface.
<img width="1843" height="812" alt="Steam Personal AI Assistant landing page" src="https://github.com/user-attachments/assets/cb1ecc2f-6c12-46d4-8400-e67bf171c845" />

[Load Your Profile]
Load a profile using a SteamID64, vanity name, or Steam Community profile URL. 
Your game library and playtime must be publicly accessible for Steam to return them.
<img width="1884" height="814" alt="Loading a Steam profile" src="https://github.com/user-attachments/assets/83d5bf32-bf63-4e87-8d44-1b3699ce06ad" />

[Browse Your Prepared Library]
Select Prepare library to augment your information with available Steam store metadata.
The library table includes game descriptions, genres, release dates, developers, publishers, and recorded playtime in hours. Games without available store details remain included in your library.
<img width="1883" height="742" alt="Prepared Steam library table" src="https://github.com/user-attachments/assets/79e226be-c359-4bc4-a0c7-b5913ef85c0e" />

[Discover Your Gaming Habits]
The Insights tab provides summary cards and interactive chart selections:
- Games owned and total recorded hours.
- Unplayed games: games with zero recorded playtime.
- Briefly played games: games with more than zero but fewer than two recorded hours.
- Hours by genre: the genres accounting for the most playtime.
- Unplayed genres: the most common genres in your unplayed library.
- Playtime breakdown: game counts grouped by recorded hours.
Multi-genre games contribute to every listed genre, so genre totals overlap.
<img width="1860" height="807" alt="Library summary cards and genre insights" src="https://github.com/user-attachments/assets/8b43abb3-b1a5-4964-bb2b-d09588e3efbb" />

[Ask Your Library Assistant]
Chat with an OpenAI model about the prepared library. Ask factual questions, explore overlooked games, or request suggestions based on your interests and mood.
For example:
- “What are my top three most-played games?”
- “Suggest an unplayed game for a relaxing evening.”
- “Which games in my library have similar themes?”
- “Help me choose something I have barely played.”
Responses stream into the chat, and recent conversation history supports follow-up questions.
<img width="1891" height="786" alt="AI conversation about a Steam library" src="https://github.com/user-attachments/assets/fd7ccfa5-093e-46d8-90d1-f640587cc233" />

[How It Works]
1. Retrieve owned games and recorded playtime from the Steam Web API.
2. Fetch available store metadata and clean HTML from descriptions.
3. Combine the records into a pandas DataFrame.
4. Calculate library summaries and visualizations.
5. Send the prepared library and recent conversation to OpenAI to answer questions.
The current assistant uses GPT-4.1 mini.

[Limitations and Data Use]
Answers depend on the information Steam makes available and may contain AI-generated errors. The app does not currently retrieve achievements, purchase dates, or current prices.
Library data and recent chat messages are sent to OpenAI when you ask a question. API usage incurs costs. 
Charts in the Insights tab are calculated by the application.
[Disclaimer]
This application is not endorsed by, affiliated with, or a product of Steam or Valve Corporation. It is a fan-made project created for fun.
