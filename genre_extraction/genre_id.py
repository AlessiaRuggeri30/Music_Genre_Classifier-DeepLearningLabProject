from pprint import pprint

TRACK_FILE = "msd_genre_dataset.txt"

with open(TRACK_FILE, 'r', encoding='utf-8') as fl:
    data = fl.read()
    lines = data.split('\n')

    done = 0

    file = open("../genre_id/genre_id.txt", "w", encoding='utf-8')

    for line in lines:
        if line[0] == '#' or line[0] == ' ' or line[0] == '%':
            continue
        genre = line.split(',')[0]
        idb = line.split(',')[1]

        to_store = genre + "," + idb + "\n"

        done += 1

        file.write(to_store)
        print('Done {}'.format(done))

print("Finished!")