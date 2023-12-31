import openai

def predict(description):
    import os
    openai.api_key = os.environ["OPENAI_KEY"]

    prompt = f"""
    classify as social action or not social action following the examples below:
    
    
    "I like you. I love you", 0
"Community Workshop on Energy Efficiency: A workshop aimed at educating residents about energy-saving practices.", 1
"Celebrity Launches New Fragrance Line: A famous celebrity launches their own line of exclusive fragrances.", 0
"Charity Run for Mental Health Awareness: Organizing a charity run to raise awareness and funds for mental health.", 1
"New Action Movie Starring Renowned Actor: A renowned actor stars in a new, high-budget action movie.", 0
"Tree Planting Initiative by Local Schools: Local schools collaborate on a tree planting initiative to beautify and green their neighborhoods.", 1
"Famous Singer Releases Party Music Album: A famous singer releases a new album focused on upbeat party music.", 0
"Soup Kitchen Expansion to Serve More Homeless: A local soup kitchen expands its facilities to serve more homeless individuals.", 1
"Top Model Debuts High Fashion Clothing Line: A top model debuts their own high fashion clothing line in a glamorous event.", 0
"Book Drive for Rural Communities: Initiating a book drive to provide educational materials to underfunded rural schools.", 1
"Celebrity Chef Opens Luxury Restaurant: A celebrity chef opens a new luxury restaurant in a trendy neighborhood.", 0
"Free Medical Camp for Underserved Areas: Organizing a free medical camp to provide healthcare services in underserved areas.", 1
"Movie Star Invests in Exclusive Beach Resort: A movie star invests in developing an exclusive beach resort.", 0
"Community Recycling Awareness Campaign: A campaign to promote recycling and sustainability within the community.", 1
"Pop Star's Lavish Island Vacation: A pop star's lavish vacation on a private island is featured in entertainment news.", 0
"Neighborhood Initiative for Cleaner Streets: A community-led initiative to clean and maintain local streets and public areas.", 1
"Famous Athlete Launches Sports Apparel Brand: A famous athlete launches their own line of sports apparel and accessories.", 0
"Workshop for Sustainable Urban Farming: Conducting a workshop on sustainable urban farming practices for city dwellers.", 1
"Rock Band's World Tour Announcement: A popular rock band announces dates for their upcoming world tour.", 0
"Charity Auction for Wildlife Conservation: Hosting a charity auction to raise funds for wildlife conservation efforts.", 1
"Actress Debuts in Broadway Play: A well-known actress makes her debut in a highly anticipated Broadway play.", 0
"Fundraiser for Local Homeless Shelter: Organizing a fundraiser to support the operations of a local homeless shelter.", 1
"Celebrity Endorses Luxury Watch Brand: A celebrity endorses a luxury watch brand in a high-profile advertising campaign.", 0
"Community Language Learning Program: Launching a language learning program to help immigrants and refugees integrate.", 1
"Famous Musician's Extravagant Birthday Bash: A famous musician throws an extravagant birthday bash with numerous celebrities.", 0
"Health Workshop for Elderly Residents: A community health workshop focusing on the needs of elderly residents.", 1
"Actress's Line of Designer Handbags: A famous actress launches her own line of designer handbags.", 0
"Volunteer Tutoring for Disadvantaged Youth: Setting up a volunteer tutoring program to assist disadvantaged youth with their studies.", 1
"Director's New Historical Drama Film: A renowned director releases a new film focusing on a dramatic historical event.", 0
"Food Bank's Drive to Stock Up for Winter: A local food bank's drive to collect food and supplies for the winter season.", 1
"TV Host's New Game Show Premiere: A popular TV host premieres their new, highly anticipated game show.", 0
"Community Fitness Program for Health Promotion: Launching a community fitness program to promote health and wellness.", 1
"Singer's Exclusive Concert on Luxury Cruise: A famous singer performs an exclusive concert on a luxury cruise ship.", 0
"Animal Shelter Volunteer Training Program: An animal shelter sets up a training program for new volunteers.", 1
"Celebrity Autobiography Release and Book Tour: A celebrity releases their autobiography and announces a book tour.", 0
"Local River Conservation Efforts: Efforts by community groups to conserve and protect a local river ecosystem.", 1
"Reality TV Star's Fashion Week Debut: A reality TV star debuts on the runway at a major fashion week event.", 0
"Health Awareness Campaign for Heart Disease: A local health organization launches a campaign to raise awareness about heart disease.", 1
"Major Update to Popular Software Suite: A tech company releases a major update to its popular office software suite.", 0
"Youth Sports Clinic for Underprivileged Children: A free sports clinic is organized for underprivileged children to promote physical activity.", 1
"Launch of a New Smartphone: A leading tech company announces the release of its latest smartphone model.", 0
"Clothing Drive for Homeless Shelters: Community members organize a clothing drive to help local homeless shelters.", 1
"New Discovery in Particle Physics: Scientists announce a significant discovery in the field of particle physics.", 0
"Community Workshop on Recycling: A workshop to educate residents about recycling and waste reduction techniques.", 1
"Global Tech Conference Announced: The dates for a major global tech conference are announced, featuring industry leaders.", 0
"Charity Walk for Diabetes Research: A charity walk is organized to raise funds for diabetes research and awareness.", 1
"Innovative Smart Home Technology Unveiled: A company unveils its innovative smart home technology designed for energy efficiency.", 0
"Fundraiser for Refugees: A fundraiser event to support refugees with essential supplies and services.", 1
"Breakthrough in Wireless Communication Tech: A tech company reveals a breakthrough in high-speed wireless communication.", 0
"Free Legal Aid Clinic for Low-Income Residents: A free legal aid clinic is set up to assist low-income residents with legal issues.", 1
"New High-End Fashion Line Released: A famous fashion designer releases a new high-end fashion line.", 0
"Environmental Film Screening and Discussion: A screening of a documentary about climate change, followed by a discussion.", 1
"Debut of a New Electric Sports Car: A car manufacturer debuts its new electric sports car model.", 0
"Community Cooking Classes for Healthy Eating: Free cooking classes are offered to teach healthy eating habits.", 1
"Virtual Reality Gaming System Launch: A gaming company launches its new virtual reality gaming system.", 0
"Book Donation Drive for Libraries: A book donation drive to support underfunded libraries and encourage reading.", 1
"New AI-driven Investment Tool Released: A financial tech company releases a new AI-driven tool for investors.", 0
"Local Theater Group Performs for Charity: A local theater group performs a play with proceeds going to a children's charity.", 1
"Advanced Robotics Exhibition: An exhibition showcasing the latest advancements in robotics technology.", 0
"Community Bicycle Repair Workshop: A free workshop teaching bicycle repair and maintenance skills.", 1
"Premiere of a New Sci-Fi Movie: The premiere of a highly anticipated new sci-fi movie.", 0
"Volunteer Firefighter Recruitment Drive: A recruitment drive for volunteer firefighters in the community.", 1
"Unveiling of a New Luxury Yacht Model: A yacht manufacturer unveils its latest luxury yacht model.", 0
"Charity Gala for Historical Preservation: A charity gala to raise funds for the preservation of historical buildings.", 1
"Launch of a Space Exploration Startup: A startup focusing on space exploration announces its official launch.", 0
"Food Festival to Support Local Farmers: A food festival that showcases and supports local farmers and producers.", 1
"New Advanced Drone Technology Revealed: A tech company reveals its new advanced drone technology for commercial use.", 0
"Workshop on Sustainable Gardening: Local experts conduct a workshop on sustainable gardening techniques for urban dwellers.", 1
"Innovations in Renewable Energy: A startup company unveils its latest innovations in solar energy technology.", 0
"Charity Event for Homeless Pets: A local animal shelter organizes a charity event to raise funds for homeless pets.", 1
"Breakthrough in Cancer Treatment: Researchers announce a new, more effective treatment for a specific type of cancer.", 0
"Community Workshop on Financial Literacy: A free workshop aiming to improve financial literacy among low-income families.", 1
"Launch of a New Luxury Car Model: A well-known car brand launches its latest luxury car model with advanced features.", 0
"School Supplies Drive for Low-Income Students: A campaign to collect and distribute school supplies to students from low-income families.", 1
"New Findings in Deep Space Exploration: Scientists reveal groundbreaking findings from a recent deep space exploration mission.", 0
"Neighborhood Clean Air Initiative: A community-led initiative to reduce air pollution in the neighborhood.", 1
"Tech Giant Unveils New Gaming Console: A major tech company releases its latest gaming console with advanced graphics and features.", 0
"Latest iPhone Model Released: Apple has just released its latest iPhone model with new features and improvements.", 0
"Community Rallies for Affordable Housing: Citizens and activists gather to discuss strategies for promoting affordable housing in the city.", 1
"Breakthrough in Alzheimer's Research: Scientists announce a significant breakthrough in the treatment of Alzheimer's disease.", 0
"Local River Cleanup Initiative: Volunteers from various neighborhoods come together to clean the polluted riverbanks.", 1
"New Horizons in Space Travel: A private company unveils its plan to send tourists into space by next year.", 0
"Fundraiser for Local Animal Shelter: A fundraiser is organized to support the local animal shelter struggling with increased intake.", 1
"Advancements in Electric Vehicle Technology: A major car manufacturer announces new battery technology, increasing electric vehicle range.", 0
"Protest Against Deforestation: Environmental activists organize a protest against the deforestation in the Amazon rainforest.", 1
"Discovering the Depths of the Ocean: Marine biologists discover new marine species in a previously unexplored part of the ocean.", 0
"Community Free Health Check-up Camp: A free health check-up camp is set up by a local hospital to serve underprivileged communities.", 1
"Innovations in Artificial Intelligence: Tech company reveals a new AI system that can predict weather patterns with high accuracy.", 0
"Rising Concerns Over Privacy Breaches: Recent reports highlight increasing incidents of personal data breaches from major tech companies.", 0
"Neighborhood Solar Panel Project: A community initiative is launched to install solar panels in the neighborhood, promoting renewable energy use.", 1
"Controversy Over New Social Media Algorithm: Debate arises over a social media platform's new algorithm, which is accused of promoting divisive content.", 0
"Charity Marathon for Cancer Research: A city-wide marathon is organized to raise funds and awareness for cancer research.", 1
"Expansion of Surveillance Technologies: Government expands the use of surveillance technologies in public spaces, raising privacy concerns.", 0
"Local Artists Collaborate for Charity Exhibition: Artists in the community come together to host an exhibition, with proceeds going to the homeless shelter.", 1
"Debate Over Nuclear Energy Expansion: Public forums are being held to discuss the potential expansion of nuclear energy and its environmental impact.", 0
"Free Coding Workshops for Youth: Local tech companies are offering free coding workshops to encourage digital skills among underprivileged youth.", 1
"Controversial Land Development Plan: A new land development plan is facing opposition due to potential environmental damage.", 0
"Community Garden Project for Urban Areas: Residents in urban areas are coming together to create community gardens, promoting green spaces.", 1
"Rising Concerns Over Deepfake Technology: Experts warn about the increasing sophistication of deepfake technology and its potential misuse.", 0
"Beach Accessibility Project: A new initiative is launched to improve beach accessibility for people with disabilities.", 1
"City Implements Strict Recycling Laws: New recycling regulations aim to reduce waste, but some residents and businesses express concerns over the cost.", 0
"Fundraiser for Disaster Relief Efforts: A local NGO organizes a fundraiser to aid victims of recent natural disasters.", 1
"Debate Over Genetically Modified Crops: The use of genetically modified crops sparks debate among farmers, scientists, and consumers.", 0
"Youth Mentorship Program Launch: A new program pairs at-risk youth with mentors to help them achieve their academic and personal goals.", 1
"Concerns Over Facial Recognition Use: The use of facial recognition technology by law enforcement agencies raises privacy and ethics concerns.", 0
"Community Literacy Campaign: Volunteers band together to improve literacy rates among children and adults in underserved communities.", 1
"Expansion of Urban Surveillance: The city's decision to expand surveillance cameras in public areas is met with mixed reactions.", 0
"Local River Restoration Project: Environmental groups and volunteers work to restore the health of a polluted river.", 1
"Controversy Over High-Speed Rail Project: The proposed high-speed rail project faces opposition due to its environmental impact and cost.", 0
"Neighborhood Emergency Preparedness Workshop: Residents participate in a workshop to prepare for natural disasters and emergencies.", 1
"Increased Screen Time Affects Children's Health: Studies show that excessive screen time is negatively impacting children's physical and mental health.", 0
"Community Organizes Beach Cleanup: Local residents and environmentalists are coming together this weekend to clean up the shoreline at Brighton Beach.", 1
"Advances in Quantum Computing: Researchers at MIT have made a breakthrough in quantum computing speeds.", 0
"Fundraiser for Local Schools: A charity event is being organized to raise funds for new equipment in local schools.", 1
"New Species of Frog Discovered: A new species of frog has been discovered in the Amazon rainforest.", 0
"Volunteers Needed for Food Drive: The city food bank is seeking volunteers for their weekly food drive to help the underprivileged.", 1
"Tech Company Launches New Smartphone: A leading tech company has announced the release of its latest smartphone model.", 0
"Neighborhood Watch Program Expands: The local neighborhood watch is expanding to include more areas for community safety.", 1
"Breakthrough in Renewable Energy: Scientists have developed a more efficient solar panel.", 0
"Book Drive for Underfunded Libraries: A campaign is underway to collect books for underfunded libraries in rural areas.", 1
"Rare Bird Spotted in National Park: Birdwatchers have reported sightings of a rare bird in Yellowstone National Park.", 0
"City to Plant 1000 Trees: A new initiative aims to plant 1000 trees in urban areas.", 1
"Innovative Water Purification Method Developed: Scientists have created a low-cost water purification system.", 0
"Local Artists Host Free Workshop: A group of local artists is offering free workshops to the community.", 1
"Discovery of Ancient Ruins: Archaeologists have discovered ancient ruins in Greece.", 0
"Youth Soccer League Kicks Off: A new youth soccer league starts this weekend, promoting sports among children.", 1
"AI Predicts Weather Patterns More Accurately: A new AI system can predict weather patterns with greater accuracy.", 0
"Community Garden Expansion: The community garden is expanding to allow more local residents to grow their own food.", 1
"Breakthrough in Alzheimer's Research: Scientists have identified a new target for Alzheimer's treatment.", 0
"Free Coding Classes for Teens: A tech group is offering free coding classes to teenagers.", 1
"New Dinosaur Species Unearthed: Paleontologists have discovered a new species of dinosaur in Argentina.", 0
"Recycling Program Success: The city's recycling program has significantly reduced waste.", 1
"New Galaxy Observed Through Telescope: Astronomers have observed a previously unknown galaxy.", 0
"Community Health Fair This Weekend: A health fair offering free screenings will take place this weekend.", 1
"Major Leap in Battery Technology: A new battery technology promises longer life for electronic devices.", 0
"Neighborhood Clean-Up Day Scheduled: Residents are encouraged to participate in the upcoming neighborhood clean-up day.", 1
"Revolutionary New Antibiotic Discovered: Researchers have discovered a new antibiotic that fights resistant bacteria.", 0
"Local Library Extends Hours: The local library will now be open longer to serve the community better.", 1
"New Method for Plastic Recycling: A breakthrough in plastic recycling could drastically reduce pollution.", 0
"Volunteer Firefighters Honored: The community is honoring its volunteer firefighters with a special ceremony.", 1
"Discovery of a New Comet: Astronomers have discovered a new comet passing close to Earth.", 0
"Black Friday: seven things to do instead of buying stuff Words by Gavin Haines November 2",1
"a startup is creating a concept to turn poverty into history",1 
"New Study Reveals Link Between Social Media Use and Mental Health: Researchers find a correlation between social media use and increased symptoms of depression and anxiety.", 0
"Community Rallies to Support Local Food Bank: Residents come together to donate food and volunteer their time to help those in need.", 1
"Government Announces Plan to Address Climate Change: Leaders pledge to take action to reduce carbon emissions and mitigate the effects of climate change.", 0
"Nonprofit Organization Provides Education and Resources for Refugees: Organization works to help refugees access education and resources to improve their lives.", 1
"Local Businesses Partner to Support Community Projects: Business owners collaborate to fund and support initiatives that benefit the local community.", 0
"Experts Warn of Cybersecurity Threats: Security professionals caution about the increasing number of cyberattacks and the need for greater protection measures.", 1
"New App Aims to Improve Mental Health Support: Developers launch an app that provides mental health resources and support to users.", 0
"Community Comes Together to Support LGBTQ+ Rights: Activists and allies rally to demand equal rights and protections for the LGBTQ+ community.", 1
"Government Launches Initiative to Address Homelessness: Leaders announce a new program aimed at providing housing and support services to those experiencing homelessness.", 0
"Researchers Discover New Species of Plant: Scientists identify a previously unknown species of plant and work to protect it from extinction.", 1
"Study Reveals Link Between Social Media Use and Mental Health: Researchers find that excessive social media use can lead to increased symptoms of depression and anxiety.", 0
"Community Rallies to Support Local Food Bank: Residents come together to donate food and volunteer at a local food bank to help those in need.", 1
"New Law Aims to Protect Immigrant Rights: Advocates celebrate the passage of a new law that protects the rights of immigrants and refugees.", 1
"School Program Teaches Children About Climate Change: Educators launch a new program to educate children about climate change and its impact on the environment.", 0
"City Council Votes to Increase Minimum Wage: Lawmakers pass a bill to increase the minimum wage in the city to help low-income workers.", 1
"Nonprofit Organization Provides Free Legal Services to Low-Income Families: A new nonprofit organization launches a program to provide free legal services to low-income families in need.", 1
"Art Exhibit Highlights Social Justice Issues: Artists showcase their work at an exhibit that highlights social justice issues and promotes activism.", 0
"Community Garden Created to Promote Food Security: Residents come together to create a community garden that provides fresh produce to those in need.", 1
"New App Helps People Find Affordable Housing: Developers launch a new app that connects people with affordable housing options in their area.", 0
"Scholarship Program Provides Financial Support for Women in STEM Fields: A new scholarship program is launched to provide financial support for women pursuing degrees in science, technology, engineering, and math (STEM) fields.", 1
"City Installs Electric Vehicle Charging Stations: The city installs new electric vehicle charging stations to promote sustainable transportation and reduce carbon emissions.", 0
"Volunteer Program Helps People with Disabilities Find Employment: Organizations launch a new volunteer program that pairs people with disabilities with employers who are committed to hiring and supporting employees with disabilities.", 1
"New Law Requires Employers to Provide Paid Family Leave: Lawmakers pass a bill that requires employers to provide paid family leave to their employees.", 1
"Community Health Fair Promotes Healthy Living: Local healthcare providers host a community health fair to promote healthy living and provide free health screenings to residents.", 0
"New School Program Focuses on Financial Literacy: Educators launch a new program that teaches children about financial literacy and how to manage money responsibly.", 0
"City Partners with Local Businesses to Offer Free Wi-Fi: The city partners with local businesses to provide free Wi-Fi to residents in public spaces.", 0
"Nonprofit Organization Provides Job Training for At-Risk Youth: A new nonprofit organization launches a job training program for at-risk youth to help them gain the skills and experience they need to succeed in the workforce.", 1
"Art Exhibit Highlights the Impact of Climate Change: Artists showcase their work at an exhibit that highlights the impact of climate change and the need for sustainable solutions.", 0
"Community Garden Created to Promote Food Security: Residents come together to create a community garden that provides fresh produce to those in need.", 1
"New App Helps People Find Affordable Healthcare: Developers launch a new app that connects people with affordable healthcare options in their area.", 0


Compete the next given teh example
"{description}","""

    result = openai.chat.completions.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": "you are a classifier of social action articles"},
            {"role": "user", "content": prompt},
        ],
    )
    return int(result.choices[0].message.content)