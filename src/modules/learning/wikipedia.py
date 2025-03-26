import wikipediaapi
import re

wiki_wiki = wikipediaapi.Wikipedia(user_agent='Svasa AI/1.0 (https://www.svasa.ai) Learning Module', language='en', extract_format=wikipediaapi.ExtractFormat.HTML)

def get_page(page_title: str):
    page = wiki_wiki.page(page_title)
    if page.exists():
        return page.text
    else:
        return None
    
def get_links(page_title: str):
    page = wiki_wiki.page(page_title)
    if page.exists():
        return page.links
    else:
        return None
    
def get_url(page_title: str):
    page = wiki_wiki.page(page_title)
    if page.exists():
        return page.fullurl
    else:
        return None
    
def get_summary(page_title: str):
    page = wiki_wiki.page(page_title)
    if page.exists():
        return page.summary
    else:
        return None

def update_summary_and_links(links, summary):
    updated_links = []
    new_summary = summary
    # Sort links by descending length to replace longer phrases first
    for link in sorted(links, key=len, reverse=True):
        if link in summary:
            updated_links.append(link)
            new_summary = re.sub(re.escape(link), f"[[{link}]]", new_summary)
    return updated_links, new_summary
