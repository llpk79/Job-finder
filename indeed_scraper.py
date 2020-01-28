import os
import smtplib
import spacy
import urllib3

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from email.message import EmailMessage
from pyshorteners import Shortener
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Load environment variables.
# Docker
env_path = '/usr/src/.env'
load_dotenv(dotenv_path=env_path, verbose=True)

# Local
# load_dotenv()

PASSWORD = os.getenv('PASSWORD')
USER_NAME = os.getenv('USER_NAME')
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
API_KEY = os.getenv('API_KEY')


class IndeedScraper(object):
    """Get Indeed.com job listings that match closest with your provided document.
    
    Use BeautifulSoup4 to scrape Indeed.com for job listings and descriptions.
    Use the Spacy NLP library to vectorize each listing and your provided document.
    Then, use KNN to find listings most relevant to your provided document.
    Get the results right in your email inbox!

    User supplies:
    - The number of indeed.com pages to search.
    - The number of search results to return.
    - The email address to send results to.
    - The file name of your provided document.
    - The city you want to work in.
    - The state that city is in.
    - A search term for the kind of job you're looking for (i.e. Data Scientist).

    """

    def __init__(self) -> None:
        self.pages = self.num_user_input('\nEnter number of pages to search:\n')  # Number of indeed pages to search.
        self.num_jobs = self.num_user_input('\nEnter max job listings to receive:\n') * 2  # Buffer for duplicates.
        self.resume = self.load_resume()
        self.email = self.user_input('\nEnter email:\n')
        print('\nYou may leave any of the following prompts blank to broaden your search.')
        self.city = self.user_input('\nEnter desired city:\n').strip().title()
        self.state = self.user_input('\nEnter state abbreviation:\n').strip().upper()
        self.terms = self.user_input('\nEnter desired job title:\n').strip().lower()
        self.url = self.build_url()
        self.http = urllib3.PoolManager()
        print('\nLoading NLP packages...')
        self.nlp = spacy.load('en_core_web_lg')
        self.nn = NearestNeighbors(n_neighbors=self.num_jobs,
                                   algorithm='ball_tree')
        self.shortener = Shortener('Bitly', bitly_token=API_KEY)
        self.jobs = []
        self.base_email = EMAIL_ADDRESS
        self.vectors = None
        self.descriptions = None
        self.main()
    
    def main(self) -> None:
        """Calls all methods needed to complete program."""
        self.descriptions = self.get_descriptions()
        self.vectors = self.get_description_vectors()
        self.get_best_jobs()
        self.email_jobs()

    def load_resume(self) -> str:
        """Load resume text from disc."""
        while True:
            path = self.user_input('\nEnter document file name:\n')
            if path[-3:] != "txt":
                print(f'\n{"-" * 20}')
                print('File name must end in ".txt"')
                print(f'{"-" * 20}')
                continue
            try:
                with open(path, 'r') as f:
                    print('\nLoading document...')
                    resume = f.read().strip('\n')
                break
            except FileNotFoundError:
                print(f'\n{"-" * 20}')
                print(f"Can't find {path}")
                directory = os.path.curdir
                files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
                print(f'Files I can see: \n\n{", ".join(files)}\n')
                print('Did you copy your file to the Docker container?')
                print('See instructions at: https://github.com/llpk79/Job-finder/packages/111889')
                print(f'{"-" * 20}')
        return resume

    def build_url(self) -> str:
        """Builds search url from user input."""
        url = f'http://www.indeed.com/jobs?q=' \
              f"{'%20'.join(self.terms.split())}&l={'%20'.join(self.city.split())},%20{self.state}"
        print(f'\nSearch URL: {url}')
        return url

    @staticmethod
    def user_input(prompt: str) -> str:
        """Prompts user with <prompt> and returns string input."""
        return input(prompt)

    @staticmethod
    def num_user_input(prompt: str) -> int:
        """Prompts user with <prompt> and returns integer input."""
        while True:
            try:
                num = int(input(prompt))
                return num
            except ValueError:
                print('\nPlease enter a number.\n')
                continue

    @staticmethod
    def find_long_descriptions(soup) -> list:
        """Create list of urls for long form job descriptions."""
        urls = []
        for div in soup.find_all(name='div', 
                                 attrs={'class': 'row'}):
            for a in div.find_all(name='a', 
                                  attrs={'class': 'jobtitle turnstileLink'}):
                urls.append(a['href'])
        return urls

    def get_next_pages(self) -> list:
        """Create a list of top level pages to search."""
        return [self.url] + [self.url + f'&start={x}0' for x in range(1, self.pages)]

    def get_descriptions(self) -> list:
        """Create a list of text extracted from long form job descriptions."""
        print('\nGetting job descriptions...\n')
        descriptions = []
        # Get and parse each top level page.
        for base_url in tqdm(self.get_next_pages()):
            request = self.http.request('GET',
                                        base_url)
            base_soup = BeautifulSoup(request.data, 'html.parser')
            # Follow links to each job description on the page.
            for url in self.find_long_descriptions(base_soup):
                the_url = "http://www.indeed.com/" + url
                req = self.http.request('GET', 
                                        the_url,
                                        headers={'User-Agent': 'opera'},
                                        retries=urllib3.Retry(connect=500, 
                                                              read=2,
                                                              redirect=50))
                # Parse out title and text from each description page and put it in the descriptions list.
                soup = BeautifulSoup(req.data, 'html.parser')
                title = soup.find(name='h3',
                                  attrs={'class': 'icl-u-xs-mb--xs icl-u-xs-mt--none jobsearch-JobInfoHeader-title'})
                description = soup.find(name='div', 
                                        attrs={'id': 'jobDescriptionText'})
                if description:
                    descriptions.append((the_url, description.text, title.text))
        print(f"\nFound {len(descriptions)} jobs.")
        return descriptions

    def get_description_vectors(self) -> list:
        """Get Spacy vectors for each long form job description."""
        print('\nGetting description vectors...\n')
        return [self.nlp(doc).vector for _, doc, _ in tqdm(self.descriptions)]
        
    def get_best_jobs(self) -> None:
        """Vectorize resume and fit a nearest neighbors classifier to find desired number of jobs."""
        print(f'\nFinding best {self.num_jobs // 2} job matches...\n')
        self.nn.fit(self.vectors)
        neighbors = list(self.nn.kneighbors([self.nlp(self.resume).vector], self.num_jobs, return_distance=False)[0])
        for neighbor in neighbors:
            self.jobs.append(self.descriptions[neighbor])
        self.remove_duplicates()

    def remove_duplicates(self) -> None:
        """Use Spacy's similarity function to weed out duplicate job descriptions."""
        final_jobs = [self.jobs[0]]
        for job in tqdm(self.jobs[1:]):
            doc1 = self.nlp(job[1])
            # Compare the similarity of <job> to each doc in <final_jobs>.
            # If <job> matches any of <final_jobs> reject <job>.
            if all([doc1.similarity(self.nlp(doc[1])) < .99 for doc in final_jobs]):
                final_jobs.append(job)
                # Don't include more jobs than were asked for.
                if len(final_jobs) == self.num_jobs // 2:
                    break
        self.jobs = final_jobs.copy()

    def email_jobs(self) -> None:
        """Send list of jobs to user."""
        print('\nEmailing jobs...\n')
        msg = self.build_message()
        server = self.initialize_server()
        self.send_and_deactivate(server, msg)

    def build_message(self) -> EmailMessage:
        """Create EmailMessage instance."""
        msg = EmailMessage()
        msg['subject'] = 'New jobs!!'
        msg['from'] = self.base_email
        msg['to'] = self.email
        div = '\n' + '*-' * 20 + '\n'
        msg.set_content(f'{div}'.join([job[2] + '\n\n' + self.shortener.short(job[0]) + '\n\n' + job[1] + '\n\n'
                                       for job in self.jobs]))  # job == (url, description, title)
        return msg

    @staticmethod
    def initialize_server() -> smtplib.SMTP:
        """Start a Gmail smtp server."""
        server = smtplib.SMTP('smtp.gmail.com', 587)
        # server.set_debuglevel(1)
        server.starttls()
        server.login(USER_NAME, PASSWORD)
        return server

    @staticmethod
    def send_and_deactivate(server: smtplib.SMTP, msg: EmailMessage) -> None:
        """Send <msg> and deactivate <server>. Print confirmation message."""
        server.send_message(msg)
        server.quit()
        print("You've got mail!!")


if __name__ == "__main__":
    scraper = IndeedScraper()
