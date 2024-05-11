# DM Project: Mapping Movie Magic: Actor Alchemy and Genre Genesis in CMU Movie & Oscars Data

## Project Abstract
This endeavour endeavours to unveil latent (rather, implicit) intricacies within the cinematic landscape by delving into actor personas and genre dynamics. I’ve strived to employ concepts taught in this course, other courses at Plaksha as well as some self-taught stuff like network analysis, machine learning, and natural language processing. This research probes the intricate web of connections moulding actor trajectories, discerns prevailing genre archetypes and evolutionary patterns, and identifies factors influencing box office success. The overarching aim is to garner newer, more profound insights into how actors traverse roles within collaborative frameworks, the influence of network centrality on career longevity, and the ebb and flow of genre preferences over time. Moreover, this study endeavours to ascertain the impact of Oscar recognition on a film's financial performance and to explore the role of regional and linguistic nuances in shaping genre predilections. Through this comprehensive exploration, I’ve tried to shed light on the multifaceted pathways shaping the evolution of the cinema.Additionally, the project examines the influence of Oscar recognition on a film's financial performance and the role of regional/linguistic nuances in shaping genre preferences. Through this comprehensive analysis, I aim to shed light on the multifaceted pathways shaping the evolution of cinema.

## Data Analysis Questions
1. What role do regional and linguistic differences play in shaping genre preferences? (Top 5 countries and Top 10 genres have been considered)
2. How can we use machine learning techniques to group movies based on their genres, languages, and release years into distinct clusters?
3. Can we make use of NLP to determine the Genre of a Movie from the Movie's Name?
4. How does network centrality differ between male and female actors over time, and does this disparity influence the types of roles or genres they are offered?
5. How does a movie's genre influence the likelihood of winning Oscars in different categories (e.g., Cinematography, Directing, Film Editing, Music, Best Picture, Writing)?

## Additional dataset used
The Oscars dataset (for question 5)

## Data Preprocessing and Initial Analysis

After importing the necessary libraries, we load three distinct datasets into pandas DataFrames in Python. The first dataset, 'movie.metadata.tsv', contains information about movies such as their Movies and Freebase IDs, release date, box office revenue, runtime, languages, countries, and genres. The second dataset, 'character.metadata.tsv', provides details about characters including their Wikipedia and Freebase IDs, release date, names, actors' date of birth, gender, height, ethnicity, and age at the time of movie release. Finally, the third dataset, 'plot_summaries.txt', furnishes plot summaries indexed by their Movie IDs. 

### Datasets
1. **Movie Metadata**: Contains information like Movie IDs, release date, box office revenue, languages, and genres.
2. **Character Metadata**: Details about characters, including their Movie IDs, release date, names, actor attributes (DOB, gender, etc.).
3. **Plot Summaries**: Plot summaries indexed by Movie IDs.

Afterwards, I've run a series of commands like .info() and .describe() to get a feel of the dataset and how it looks like. Some example outputs:
![image](https://github.com/kautilya123/DM_Project/assets/114575117/60973062-b9fa-48c5-a1b7-caf3503b3749)
![image](https://github.com/kautilya123/DM_Project/assets/114575117/d930f0ff-f45b-41b0-9805-60b61b18c2d5)

After this, I’ve run a series of commands to check the null values in the datasets and delete the rows with null values as part of the data cleaning:

Before:

![image](https://github.com/kautilya123/DM_Project/assets/114575117/c1b89d0e-a0f0-4462-91da-b3680a1f4f08)

After:

![image](https://github.com/kautilya123/DM_Project/assets/114575117/86e4c518-e8cc-4aa6-97fe-26f52b89fcc5)

Some self-explanatory elementary analysis I’ve performed to get a better understanding of dataset:
![image](https://github.com/kautilya123/DM_Project/assets/114575117/db10b982-9a33-42c0-81ab-daaad62df5c0)
![image](https://github.com/kautilya123/DM_Project/assets/114575117/f69088ff-0915-4515-9278-caac72d45af6)

Finally, I merged the datasets, checked again for null values, and then made it more readable using a function called 'parse_and_join_values' to each of these columns, which attempts to parse JSON-formatted strings and join the values into a single string separated by commas.

A brief screenshot of the final merged dataset:
![image](https://github.com/kautilya123/DM_Project/assets/114575117/21b6db06-9836-4469-9dff-9299c26a0ef3)

## Approaches and Results
### Q1: What role do regional and linguistic differences play in shaping genre preferences? (Top 5 countries and Top 10 genres have been considered)
- **Approach**: Firstly, I pinpointed the top 10 genres and top 5 countries with the highest movie counts. Then, I filtered the dataset to focus solely on movies matching both these criteria. Next, I organized the data into a pivot table grouping movies by country and genre. Finally, I visualized the findings using a stacked bar chart, allowing for a straightforward comparison of genre preferences across different regions. Like this, I’ve tried to uncover any easily, visually discernible patterns influenced by regional and linguistic factors.
- **Output**:
  ![image](https://github.com/kautilya123/DM_Project/assets/114575117/49778cbe-74b5-4382-9487-2bf844d591a4)

- **Result**: As you can see, primarily, the top 5 countries are anglophones. Almost all these films are English-language films, which seem to have absolutely dominated the market. The US, ie, Hollywood, has a really good grip on cinema in the world. Leaving the visulaization for 'Invalid Data Format', it's also evident that dramas, thriller dramas and indide dramas are really prevalent both domestically in the US (from the viewpoint of Hollywood) as well as internationally.

### Q2: How can we use machine learning techniques to group movies based on their genres, languages, and release years into distinct clusters?
- **Approach**: To analyze and categorize movies based on their genre, language, and release year, I employed a systematic approach. Initially, I extracted the release year from the movie release dates and performed one-hot encoding on genres and languages, converting categorical data into numerical format. Subsequently, I standardized the dataset to ensure uniformity in feature scales. Employing the KMeans clustering algorithm with five clusters, I grouped movies according to their standardized features, aiming to uncover underlying patterns or similarities. Further, I identified unique movie names within each cluster to understand how movies were grouped based on these features. This methodology facilitates the exploration of potential relationships and trends among movies, shedding light on distinct clusters representing different movie characteristics.
-  **Output**:
   Cluster 0:
['Eastern Promises' "Charlie Wilson's War" 'Iron Man' 'Umrao Jaan'
 'Batman Begins' 'My Beautiful Laundrette' 'Syriana' 'Monsoon Wedding'
 'The Kite Runner' 'Khuda Ke Liye' 'A Mighty Heart' 'Vertical Limit']

Cluster 1:
['Pieces' 'Jaws: The Revenge' 'Hush… Hush, Sweet Charlotte' 'Deepstar Six'
 'Two Evil Eyes' 'Wait Until Dark' 'Darkness' 'The Evil Dead' 'Case 39'
 'Paranormal Activity 2' 'Henry: Portrait Of A Serial Killer' 'Feast'
 'Saturday the 14th' 'Ghost in the Machine' 'Doom' 'Black Swan'
 "Heaven's Prisoners" 'From Hell'
 'The Texas Chainsaw Massacre: The Beginning' 'Babysitter Wanted'
 'The Grudge 3' 'Malevolence' 'Chaos' 'The Haunting in Connecticut'
 'Blink' 'Bless the Child' 'The Frighteners' 'Megiddo: The Omega Code 2'
 'One Missed Call' 'Long Time Dead' 'The Invisible'
 'Godzilla vs. Megaguirus' 'The Grudge' 'From Beyond' 'Dog Soldiers'
 'Zombieland' 'Ravenous' 'Insidious'
 'Leatherface: Texas Chainsaw Massacre III' 'Psycho' 'Mimic' 'Boogeyman'
 'P2' 'Deep Blue Sea' 'Dead of Winter' 'Flatliners' 'Passengers'
 'Funny Games U.S.' 'Amusement' 'Innocent Blood' 'The Naked Jungle'
 'Saw VII' 'An American Werewolf in Paris' 'Night of the Living Dead 3D'
 'The Ninth Gate' 'Severance'
 'A Nightmare on Elm Street 5: The Dream Child' 'Dead Alive' 'Gothika'
 'Scary Movie 3' 'Nomads' 'Raising Cain' 'Near Dark'
 'Texas Chainsaw Massacre: The Next Generation' 'The Uninvited'
 'Beyond the Door' 'Witchboard' '30 Days of Night' 'Dead Heat'
 'Dominion: Prequel to the Exorcist' 'Untraceable' 'Frankenstein Unbound'
 'Cry Wolf' 'The Most Dangerous Game' 'Critters' 'Army of Darkness'
 'Sweeney Todd: The Demon Barber of Fleet Street'
 'The Midnight Meat Train' 'Hellboy' 'Hereafter'
 "The Black Waters of Echo's Pond" 'Halloween Resurrection' 'The Ring Two'
 'See No Evil' 'Below' "He Knows You're Alone" 'Exorcist: The Beginning'
 'Deadly Friend' 'Scream 4' 'Jurassic Park III' 'The Sixth Sense'
 'Halloween' 'The Hand' 'Tales from the Darkside: The Movie'
 'The Happening' 'Saw V' 'The Passion of the Christ' 'Cabin Fever'
 'Deep Rising' 'Attack of the 50 Foot Woman' 'Scream 3'
 'Elvira: Mistress of the Dark' 'The Amityville Horror' 'Joy Ride'
 'House of Wax' 'Wrong Turn 2: Dead End' 'The Company of Wolves'
 'Eden Lake' 'Jason X' 'The Hitcher' "The Devil's Backbone"
 'Friday the 13th' 'AVP: Alien vs. Predator' 'The Birds' 'Venom'
 'BloodRayne' 'Brotherhood of the Wolf' 'May' 'The Lawnmower Man'
 'A Nightmare on Elm Street 4: The Dream Master' 'Lifeforce'
 'My Soul to Take' 'Slither' "Jack's Back" 'Of Unknown Origin' 'Shocker'
 'House of 1000 Corpses' 'Whisper' 'Species II' 'Let Me In' 'Halloween 6'
 'Scary Movie' "The Astronaut's Wife" 'Saw II' 'Deadtime Stories'
 'I Know What You Did Last Summer' 'Psycho III' "Jennifer's Body"
 'Hostel 2' 'Hood of Horror' 'Beetlejuice'
 'Something Wicked This Way Comes' '28 Weeks Later' 'Identity' 'Primeval'
 'Species' 'Disturbing Behavior' 'The Crazies' 'When a Stranger Calls'
 'The Covenant' 'The Human Centipede' 'Kaboom' 'Pumpkinhead'
 'Creature from the Black Lagoon' 'The Hidden' 'A Nightmare on Elm Street'
 'Jaws 2' 'Dead Ringers' 'The Texas Chain Saw Massacre' 'The Mangler'
 'Prom Night' 'The Strangers' "George A. Romero's Survival of the Dead"
 'The Orphanage' 'Hannibal' 'Damien: Omen II' 'Resident Evil: Apocalypse'
 'Asylum' 'Cat People' 'Heart of Midnight' 'Chain Letter' 'Lost Highway'
 'Black Christmas' 'Freddy vs. Jason' 'Stay' 'Red Riding Hood' 'Carriers'
 'Arachnophobia' 'Land of the Dead' 'The Serpent and the Rainbow'
 'Haute Tension' 'Orca' 'The Howling' 'Shutter Island' 'Maximum Overdrive'
 'The Alphabet Killer' 'Shadow of the Vampire' 'Pet Sematary'
 'The Faculty' 'Stay Alive' 'Solaris' 'Fright Night' "Jacob's Ladder"
 'Angel Heart' 'Book of Shadows: Blair Witch 2' 'Octane' "Pan's Labyrinth"
 'Carnosaur' 'The Descent 2' 'The Machinist' 'The Horsemen' 'Videodrome'
 'The Mummy Returns' 'Hatchet II' "Child's Play 2"
 'The Exorcism of Emily Rose' 'The Exorcist' 'The Omen'
 'Antarctic Journal' 'Maniac Cop' 'Sphere' 'Frankenstein'
 'Lady in the Water' 'Jason Goes To Hell: The Final Friday'
 'What Ever Happened to Baby Jane?' 'American Psycho'
 'AVPR: Aliens vs Predator - Requiem' 'The Collector' 'Lady in White'
 'In Dreams' 'Rogue' 'Stigmata' 'Friday the 13th: A New Beginning'
 'Vacancy' 'The Haunting' 'Friday the 13th: The Final Chapter'
 'Halloween H20: 20 Years Later' 'Bug' 'Teeth' 'A Tale of Two Sisters'
 "Child's Play" 'Saw III' 'Quarantine' 'Underworld: Rise of the Lycans'
 'Baghead' 'The Unborn' 'Eyes Wide Shut' 'Hatchet'
 'The Rocky Horror Picture Show' 'Hideaway' 'Vampire in Brooklyn'
 'The People Under the Stairs' 'Final Destination 3' 'Frailty'
 'Timber Falls' 'Sleepy Hollow' 'Seed of Chucky' 'The Cell' 'Pulse'
 'Valentine' 'After.Life' 'Lake Placid' 'Eyes of Laura Mars'
 'Altered States' 'All The Boys Love Mandy Lane' 'The Exorcist III'
 'FeardotCom' 'The Hills Have Eyes' 'Unbreakable' 'The Name of the Rose'
 'Sleepaway Camp' 'I Was a Teenage Werewolf' 'Sisters' 'The Thing'
 'Ghosts of Mars' 'Dawn of the Dead' 'The Texas Chainsaw Massacre' 'Them!'
 'I Am Legend' 'Silent Hill' 'D-Tox' 'Eight Legged Freaks'
 'Buffy the Vampire Slayer' 'The Wolfman' 'The Ruins'
 'Underworld: Evolution' 'Cujo' 'The Phantom of the Opera' 'The Dead Zone'
 'Wolf' 'Tremors' 'Jeepers Creepers' 'Psycho II'
 'Dylan Dog: Dead of Night' 'Leprechaun 2' 'From Dusk Till Dawn'
 'Hide and Seek' 'Aliens' 'The Grudge 2'
 'Halloween 5: The Revenge of Michael Myers' 'The First Power'
 'Wrong Turn' 'They' 'Spellbinder' 'Resident Evil: Afterlife' 'The Nanny'
 'Dahmer' 'Mulholland Drive' 'Captivity' 'The Hills Have Eyes 2'
 'The Mist' 'Hellraiser' 'Final Destination 5'
 'The Haunting of Molly Hartley' 'House of the Dead' 'The Jacket'
 'The Cave' "Freddy's Dead: The Final Nightmare" 'Revenge of the Creature'
 'Let the Right One In' 'The Gate' 'Diary of the Dead' 'Halloween II'
 'The Shining' 'Secret Window' 'The Skeleton Key'
 'Gremlins 2: The New Batch' "Child's Play 3" 'Jeepers Creepers II'
 'Anacondas: The Hunt for the Blood Orchid' 'Amityville 3-D'
 'Exorcist II: The Heretic' 'Tales from the Crypt presents: Demon Knight'
 'Dreamscape' 'Pandorum' 'Dr. Giggles' 'Lord of Illusions' 'C.H.U.D.'
 'Jaws 3-D' 'What Lies Beneath' 'Poltergeist' 'Manhunter' 'Taking Lives'
 'The Omega Code' 'Blood and Chocolate'
 'A Nightmare on Elm Street 3: Dream Warriors' 'Scream 2' 'Saw'
 'Antichrist' 'Naked Souls' 'Left Behind: The Movie' 'War of the Worlds'
 'Friday the 13th Part VIII: Jason Takes Manhattan' 'Final Destination 2'
 'In the Mouth of Madness' 'Premonition' 'Ginger Snaps 2: Unleashed'
 'I Still Know What You Did Last Summer' 'The Stepfather' 'Silver Bullet'
 'A Perfect Getaway' 'Urban Legend' 'Scary Movie 2' 'Jurassic Park'
 'The Legacy' 'Dead Silence' 'Re-Animator' 'Gremlins' 'Young Frankenstein'
 "Mary Shelley's Frankenstein" 'The Keep' 'Nightwatch'
 'Godzilla: Final Wars' 'Resident Evil: Extinction'
 'Halloween 4: The Return of Michael Myers' 'Shelter' 'Cloverfield'
 'Shutter' 'Possession' 'Piranha' 'The Order' 'Mindhunters' 'Repossessed'
 'The Toxic Avenger' 'Event Horizon' 'Shaun of the Dead' 'Candyman'
 'The Gift' 'Teen Wolf Too' 'Final Destination' 'Wolfen' 'Frozen'
 'High Spirits' 'Predator' '1408' 'The Host' 'Thir13en Ghosts' 'Giallo'
 'Hellraiser III: Hell on Earth' 'Thinner' 'The Hunger' 'Hannibal Rising'
 'King Kong Lives' 'Sector 7' "Rosemary's Baby" 'The House of the Devil'
 'Kingdom of the Spiders' 'The Legend of Hell House' 'Tarantula'
 '28 Days Later' 'Godzilla' 'Race with the Devil' 'The Fly' 'The Ring'
 'The Raven' 'An American Haunting' 'Blackout' 'The Fourth Kind'
 'My Bloody Valentine' 'An American Werewolf in London' 'Dressed to Kill'
 'Signs' 'Urban Legends: Final Cut' 'Deadly Blessing' 'Willard'
 'The Silence of the Lambs' 'The Relic' 'Ghost Ship' 'Alien' 'Christine'
 'The Tenant' 'Piranha 3-D' "The 'Burbs" 'Evil Dead II' 'Leprechaun'
 'The Mothman Prophecies' "John Carpenter's The Fog" 'Nightbreed'
 'White Noise: The Light' 'Omen III: The Final Conflict' 'Dracula 2000'
 'The Eye' "A Nightmare on Elm Street 2: Freddy's Revenge" 'The Fog'
 'End of Days' 'Sorority Row' 'D-War' 'Rise: Blood Hunter'
 'Darkness Falls' 'Firestarter' 'Transylvania 6-5000' 'Bubba Ho-tep'
 'The Horror Show' 'Splice' 'Snakes on a Plane' 'Fatal Attraction'
 'I Know Who Killed Me' 'Lost Souls' 'The Return' 'The Believers'
 'Suspect Zero' 'Inland Empire' 'Wicked Stepmother' 'The Swarm'
 'The Return of Swamp Thing' 'Play Misty for Me' 'Predator 2'
 'Bride of Chucky' 'Cemetery Man' 'Bats' 'Dracula: Dead and Loving It'
 'The Blair Witch Project' 'Hostel' 'Stan Helsing' 'The Others'
 'Van Helsing' 'Hollow Man' "The Devil's Advocate" 'Sister, Sister'
 'Doomsday' "Bram Stoker's Dracula" 'King Kong' 'Tales of Terror'
 'Red Dragon' 'Stepfather II']

Cluster 2:
['The Hunger Games' 'A Cry in the Dark' 'Dark Water' ... 'Satisfaction'
 'Entrapment' 'Dr. T & the Women']

Cluster 3:
['Akira' 'Steamboy' 'Tales from Earthsea' 'The Transformers: The Movie'
 'Arrietty' 'Evangelion: 2.0' 'From up on Poppy Hill'
 'Bleach: Memories of Nobody' 'Bleach: The DiamondDust Rebellion'
 'Space Truckers' 'Paprika' 'Final Fantasy: The Spirits Within'
 'All-Star Superman' 'Summer Wars']

Cluster 4:
['The League of Extraordinary Gentlemen']

- **Result**: The output reveals the clustering of movies into five distinct groups based on their shared characteristics. In Cluster 0, a diverse selection of movies is observed, including "Eastern Promises," "Iron Man," and "Batman Begins." Cluster 1 predominantly consists of horror and thriller films such as "Black Swan," "The Grudge 3," and "Saw VII." Cluster 2 features a mix of movies, including "The Hunger Games" and "Dark Water," indicating a variety of genres represented. Cluster 3 predominantly comprises animated and science fiction movies like "Akira," "Arrietty," and "Final Fantasy: The Spirits Within." Finally, Cluster 4 contains a single movie, "The League of Extraordinary Gentlemen," indicating its distinctiveness from other films in the dataset. I've seen the movie, and it's equal parts thriller, science fiction, war & drama. Sean Connery's acting was spectacular. This clustering provides insights into the diverse composition of movies based on their genres, themes, and characteristics.

### Q3: Can we make use of NLP to determine the Genre of a Movie from the Movie's Name?
- **Approach**: Initially, I transformed movie titles into a bag-of-words representation using the CountVectorizer from sklearn, excluding common English stop words to focus on meaningful words. Next, I encoded the movie genres to prepare them for classification. The dataset was then divided into training and testing sets, allocating 80% for training and 20% for testing to assess model performance on unseen data. Subsequently, I trained a Multinomial Naive Bayes classifier, known for its effectiveness with sparse data and commonly used in text classification tasks. Using this model, predictions were made on the test data, and the model's accuracy was computed to evaluate its performance. However, this didn't turn out to be very good, as you'll see in the output.
- **Output**: Model Accuracy: 28.57%.
- **Result**: This output means that the model correctly predicted the genre of the movie based on its title in roughly 28.57% of the cases in the test dataset. While an accuracy of 28.57% is better than random guessing (which would have an accuracy of around 10% for a classification task with 10 genres), the model's performance is relatively low. This could be due to various factors such as the simplicity of the model, the complexity of the task, or the limited information provided by movie titles alone.

### Q4: How does network centrality differ between male and female actors over time, and does this disparity influence the types of roles or genres they are offered
- **Approach**: Initially, I constructed a graph representation using NetworkX, capturing connections between actors co-starring in the same movies. By iteratively adding edges for actor pairs within each film, it formed a collaborative network reflecting industry collaborations. Subsequently, I computed degree centrality for each actor, measuring their prominence based on collaboration frequency. Temporal analysis was conducted by correlating actors' centrality scores with movie release years, enabling the tracking of centrality trends over time. Finally, the results were visualized through a line plot, illustrating the fluctuation of centrality between male and female actors across different years. This methodological framework aimed to uncover potential disparities in network centrality and assess their implications on the roles or genres offered to male and female actors throughout time.
- **Output**: ![image](https://github.com/kautilya123/DM_Project/assets/114575117/a1727905-e274-440e-be9e-43f9dd195b51)

- **Result**: The output clearly shows the evolution of network centrality among male and female actors from around 1920 to 2000, revealing key differences and trends in their film industry connections. Male actors consistently exhibit higher centrality, indicating more numerous and possibly more influential connections within the industry, which likely influences the diversity and prominence of roles they are offered. In contrast, female actors not only have lower centrality but also greater volatility in their centrality over the years, hinting at less stable career trajectories and possibly fewer opportunities in a variety of roles. This disparity highlights ongoing gender dynamics in film, reflecting broader industry and societal trends regarding gender roles.

### Q5: How does a movie's genre influence the likelihood of winning Oscars in different categories (e.g., Cinematography, Directing, Film Editing, Music, Best Picture, Writing)?
- **Approach**: Initially, I imported and processed two datasets: one containing Oscar awards information and another with movie genre data. Subsequently, I merged these datasets based on movie names to consolidate relevant information. Using the merged data, I engineered features by converting movie genres into dummy variables to represent their presence or absence. Following this, I created binary target variables for each Oscar category, denoting whether a movie won in that category or not. Employing logistic regression models, I trained and evaluated predictions for each Oscar category based on movie genres, utilizing classification reports and confusion matrices to assess model performance.
- **Output**:

![image](https://github.com/kautilya123/DM_Project/assets/114575117/982569ec-cb82-444b-904b-5a4d3fa33c3c)

- **Result**:
Classification Report
Precision for class 0 (not winning) is 0.99, which indicates that the model is very effective at identifying true negatives—movies that did not win in the "Directing" category.
Recall for class 0 is also 1.00, meaning the model correctly identified 100% of the movies that did not win.
The F1-score for class 0 is 0.99, indicating a high level of precision and recall balance for this class.

Precision, Recall, and F1-score for class 1 (winning) are all 0.00, which suggests that the model failed to correctly identify any true positives—movies that did win the Oscar for "Directing." The low support (84) for this class compared to the support for class 0 (7039) could be a significant factor contributing to this failure, indicating a highly imbalanced dataset.

 Confusion Matrix
True Negatives (TN): 7039 (movies correctly identified as not winners)
False Negatives (FN): 84 (movies that won but were predicted as not winners)
True Positives (TP) and False Positives (FP): 0 (indicating no movies were predicted as winners)

 Analysis
 
The model's failure to predict any winners (class 1) accurately might be attributed to class imbalance (much more non-winners than winners). Logistic regression, without proper balancing techniques or more robust classification methods, may struggle in such scenarios.
Another contributing factor might be the feature set (genre dummies). If genre alone doesn't strongly influence the likelihood of winning an Oscar in the category of Directing, the model may not perform well.
