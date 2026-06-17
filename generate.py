import requests
from simple_term_menu import TerminalMenu
from bs4 import BeautifulSoup
import os

def grabOption(name = "Select an Option", options = []):
    menu = TerminalMenu(
        options,
        title=f"{name} (↑/↓ to move, Enter to confirm):",
    )
    choice = menu.show()
    return options[choice]

def grabPage(url):
    html = requests.get(url)
    return html.text

def harvardParse(page):
    soup = BeautifulSoup(page, 'html.parser')
    result_dict = {}
    
    current_title = "" 
    
    for element in soup.find_all(['h2', 'ol']):
        if element.name == 'h2':
            current_title = element.get_text(strip=True)
        elif element.name == 'ol':
            items = [li.get_text(strip=True) for li in element.find_all('li')]
            result_dict[current_title] = items
    
    for key in list(result_dict.keys()):
        with open(os.path.join("texts", f"{key}.md"), "w") as f:
            listText = "\n".join([f"- {line}" for line in result_dict[key]])
            f.write(f"### {key}\n\n{listText}")

def main():
    options = {"harvard" : "https://www.cs.columbia.edu/~hgs/audio/harvard.html"}
    option = grabOption(options = list(options.keys()))
    url = options[option]

    page = grabPage(url)

    if option == "harvard":
        harvardParse(page)

if __name__ == "__main__":
    main()