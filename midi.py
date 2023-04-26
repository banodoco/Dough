import mido
import csv
from mido import MidiFile

def list_note_times(midi_file):
    mid = MidiFile(midi_file)
    note_times = []
    tempo = 500000  # Default MIDI tempo (microseconds per beat)

    for i, track in enumerate(mid.tracks):
        absolute_time = 0
        for msg in track:
            absolute_time += msg.time

            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type in ['note_on', 'note_off']:
                seconds = mido.tick2second(absolute_time, mid.ticks_per_beat, tempo)
                note_times.append(seconds)

    note_times.sort()
    return note_times

def export_to_csv(note_times, output_file='output.csv'):
    with open(output_file, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Note Times (in seconds)'])

        for time in note_times:
            csv_writer.writerow([time])

def main():
    midi_file = 'input.mid'
    output_file = 'output.csv'
    note_times = list_note_times(midi_file)
    export_to_csv(note_times, output_file)
    print(f"Note times exported to {output_file}")

if __name__ == "__main__":
    main()