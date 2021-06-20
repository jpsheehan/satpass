import os.path
from skyfield.api import load, wgs84
from datetime import timezone, timedelta
import numpy as np
import matplotlib.pyplot as plt
import json
import math

ts = load.timescale()


def get_iss(reload=False):
    stations_url = 'http://celestrak.com/NORAD/elements/stations.txt'
    satellites = load.tle_file(stations_url, reload=reload)
    by_name = {sat.name: sat for sat in satellites}
    satellite = by_name['ISS (ZARYA)']
    return satellite

def get_events(sat, coordinates, t0, t1, altitude=0.0):
    pos = wgs84.latlon(*coordinates)

    sorted_events = []
    tmp_aos = None
    t, events = sat.find_events(pos, t0, t1, altitude_degrees=altitude)
    for (ti, event) in zip(t, events):
        if event == 0:
            tmp_aos = ti
        elif event == 2:
            sorted_events.append((tmp_aos, ti))
    return sorted_events

def format_time(t, tz=timezone.utc):
    return t.astimezone(tz).strftime('%Y %b %d %H:%M:%S')

def annotate_point(ax, theta, r, t, tz):
    ax.plot(theta, r, marker='o')
    ax.annotate(t.astimezone(tz).strftime('%M:%S'), (theta, r), xytext=(theta, r-10))

def time_range(aos, los, N=50):
    dt_aos = aos.utc_datetime()
    dt_los = los.utc_datetime()
    step_size = ((dt_los - dt_aos)/(N-1))
    times = [ts.from_datetime(dt_aos + i * step_size) for i in range(N-1)] + [los]
    return times

def get_plot_data(satellite, location, event):
    (aos, los) = event
    times = time_range(aos, los)
    here = wgs84.latlon(*location)
    diff = satellite - here

    points = []
    for t in times:
        topocentric = diff.at(t)
        alt, az, _ = topocentric.altaz()

        r = 90 - alt.degrees
        theta = az.radians

        points.append((theta, r, t))

    return points

def plot_graphs(satellite, location, event, tz=timezone.utc):
    data = get_plot_data(satellite, location, event)
    aos, los = event
    #plt.title(format_time(aos, tz))

    filename = os.path.join("images", format_time(aos, tz) + ".png")

    if not os.path.exists("images"):
        os.mkdir("images")

    if not os.path.exists(filename):
        print("Creating", filename)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        plot_polar_graph(ax, data, tz)
        plt.savefig(filename)
        #plt.show()

        # clear memory
        plt.close()
        plt.cla()
        plt.clf()
    else:
        print("Skipping", filename)

    return filename

def plot_polar_graph(ax, data, tz=timezone.utc):
    theta, r, _t = unzip_data(data)

    ax.plot(theta, r)

    for theta, r, t in get_points_of_interest(data):
        annotate_point(ax, theta, r, t, tz)

    # fix up the plot
    ax.set_rmin(0)
    ax.set_rmax(90)
    ax.set_rgrids((30, 60), labels=('60°', '30°'))
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    ax.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=("N", "NE", "E", "SE", "S", "SW", "W", "NW"))


def unzip_data(data):
    theta = [theta for (theta, r, t) in data]
    r = [r for (theta, r, t) in data]
    t = [t for (theta, r, t) in data]
    return theta, r, t


def get_points_of_interest(data):
    theta, r, t = unzip_data(data)
    max_idx = r.index(min(r))
    points = []
    points.append((theta[0], r[0], t[0]))
    points.append((theta[max_idx//2], r[max_idx//2], t[max_idx//2]))
    points.append((theta[max_idx], r[max_idx], t[max_idx]))
    points.append((theta[3*max_idx//2], r[3*max_idx//2], t[3*max_idx//2]))
    points.append((theta[-1], r[-1], t[-1]))
    return points

class Event:

    def __init__(self, satellite, location, start, end, tz, downlink):
        self.satellite = satellite
        self.location = location
        self.start_utc = ts.utc(*start)
        self.end_utc = ts.utc(*end)
        self.tz = tz
        self.downlink = downlink
        self.passes = []
        self.epoch = self.satellite.epoch

    def __repr__(self):
        return json.dumps({
            "satellite": self.satellite.name,
            "start": format_time(self.start_utc, self.tz),
            "end": format_time(self.end_utc, self.tz),
            "location": json.loads(repr(self.location)),
            "timezone": self.tz.utcoffset(None).total_seconds(),
            "epoch": format_time(self.epoch, self.tz),
            "passes": [json.loads(repr(p)) for p in self.passes],
        })


    def process(self):
        events = get_events(self.satellite, (self.location.lat, self.location.lon), self.start_utc, self.end_utc)
        for aos, los in events:
            p = Pass(self, aos, los)
            p.create_graphs()
            self.passes.append(p)

class Pass:

    def __init__(self, event, aos, los):
        self.event = event
        self.aos = aos
        self.los = los
        self.duration = los.utc_datetime() - aos.utc_datetime()
        self.graphs_polar = None

    def __repr__(self):
        return json.dumps({
            "aos": format_time(self.aos, self.event.tz),
            "los": format_time(self.los, self.event.tz),
            "duration": self.duration.total_seconds(),
            "image_path": self.graphs_polar
        })

    def create_graphs(self):
        self.graphs_polar = plot_graphs(self.event.satellite, (self.event.location.lat, self.event.location.lon), (self.aos, self.los), self.event.tz)

class Location:
    def __init__(self, lat, lon, city, country):
        self.lat = lat
        self.lon = lon
        self.city = city
        self.country = country

    def __str__(self):
        return f"{self.city}, {self.country}"

    def __repr__(self):
        return json.dumps({
            "lat": self.lat,
            "lon": self.lon,
            "city": self.city,
            "country": self.country
        })

def generate_report(event, output):
    with open(output, "w") as f:
        f.write("<html>")
        f.write("""<head>
                <title>ARISS SSTV Event Details</title>
                <link rel="stylesheet" type="text/css" href="style.css" />
                </head>""")
        f.write("<body>")

        f.write("<main>")
        f.write("<h1>ARISS SSTV Event Details</h1>")

        offset_secs = event.tz.utcoffset(None).total_seconds()
        pos = offset_secs >= 0
        hours = math.floor(abs(offset_secs) / 60 / 60)
        mins = int(abs(offset_secs) - hours * 60 * 60)
        offset = str(hours).rjust(2,'0') + str(mins).rjust(2,'0')
        if pos:
            offset = "+" + offset
        else:
            offset = "-" + offset
        f.write(f"<p><em>Note: All times are referenced to UTC {offset} hours. All locations are referenced to {event.location.city}, {event.location.country} ({event.location.lat}, {event.location.lon}).</em></p>")
        #f.write(f"<p><em>This data has been generated automatically and may contain errors. Download the source from <a href='https://foo.nz/'>https://foo.nz/</a>.</em></p>")
        f.write("<p><em>This document was created using ZL3JPS's satpass software.</em></p>")
        f.write(f"<p>The epoch for the trajectory data is {format_time(event.epoch, event.tz)}. This data will become unreliable if a prediction is made too far in the future (several days) from this date.")
        f.write(f"<p>The event starts at {format_time(event.start_utc, event.tz)} and ends at {format_time(event.end_utc, event.tz)}. The downlink frequency from the International Space Station is {event.downlink:.3f} MHz FM.</p>")
        f.write(f"<p class='break-column'>There are {len(event.passes)} passes in this timespan.</p>")

        for i, p in enumerate(event.passes):
            f.write("<div class='pass'>")
            f.write(f"<h3>Pass {i+1}</h3>")
            f.write(f"<p class='epoch'>as at {format_time(event.epoch, event.tz)}</p>")
            total_secs = p.duration.total_seconds()
            minutes = math.floor(total_secs / 60.0)
            seconds = str(int(total_secs - minutes * 60)).rjust(2,'0')
            minutes = str(minutes).rjust(2,'0')
            f.write(f"<p class='details'><strong>AOS:</strong> {format_time(p.aos, event.tz)}<br /><strong>LOS:</strong> {format_time(p.los, event.tz)}<br /><strong>Duration:</strong> {minutes + ':' + seconds}</p>")
            f.write(f"<img src='{p.graphs_polar}' />")
            f.write("<p class='notes'>Notes:</p>")
            f.write("</div>")

        f.write("</div>")
        f.write("</body>")
        f.write("</html>")

def main():
    START_OF_EVENT = (2021, 6, 21, 9, 40)
    END_OF_EVENT = (2021, 6, 26, 18, 30)
    CHRISTCHURCH = Location(-43.53189984688002, 172.63925976596593, "Christchurch", "New Zealand")
    iss = get_iss(reload=True)
    tz = timezone(timedelta(hours=+12))
    sstv_event = Event(iss, CHRISTCHURCH, START_OF_EVENT, END_OF_EVENT, tz, 145.800)
    sstv_event.process()

    with open("sstv_event.json", "w") as f:
        f.write(repr(sstv_event))

    generate_report(sstv_event, "report.html")

if __name__ == '__main__':
    main()

