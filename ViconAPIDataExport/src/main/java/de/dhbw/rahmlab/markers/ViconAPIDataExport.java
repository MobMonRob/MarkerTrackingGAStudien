/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */
package de.dhbw.rahmlab.markers;

import de.dhbw.rahmlab.vicon.datastream.api.DataStreamClient;
import de.dhbw.rahmlab.vicon.datastream.api.Version;
import de.dhbw.rahmlab.vicon.datastream.impl.StreamMode_Enum;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 *
 * @author rahm-lab
 */
public class ViconAPIDataExport {

    public static void main(String[] args) {

        String hostname = "192.168.10.1:801";

        DataStreamClient client = new DataStreamClient();

        Version version = client.getVersion();
        System.out.println("Version: " + version.getMajor() + "." + version.getMinor() + "." + version.getPoint());

        System.out.println("Try to connect to: " + hostname);
        client.connect(hostname, 4000l);

        client.enableMarkerData();
        client.enableCentroidData();
        client.enableUnlabeledMarkerData();

        client.setStreamMode(StreamMode_Enum.ClientPull);

        System.out.println("Marker Data Enabled:" + client.isMarkerDataEnabled());
        System.out.println("Centroid Data Enabled: " + client.isCentroidDataEnabled());
        System.out.println("Unlabeled Marker Data Enabled: " + client.isUnlabeledMarkerDataEnabled());
        
        //idk why this fetching of 2 frames is necessary. by only fetching one frame it will return 0 cameras and no results
        client.getFrame();
        System.out.println("Frame number:" + client.getFrameNumber());
        client.getFrame();
        System.out.println("Frame number:" + client.getFrameNumber());

        String dump = csvCentroidDump(client);
        System.out.println(dump);
        
        ArrayList<String> objects = new ArrayList();
        objects.add("Wand");
        objects.add("large-4-marker");

        String markerDump = csvTrackingObjectDump(client, objects);
        System.out.println("------");
        System.out.println(markerDump);
        
        System.out.println("Unlabeled Markers: " + client.getUnlabeledMarkerCount());
        System.out.println("Labeled Markers for 'Wand': " + client.getMarkerCount("Wand"));
        System.out.println("Labeled Markers for 'large-4-marker': " + client.getMarkerCount("large-4-marker"));
        //String markerDump = csvUnlabeledMarkerDump(client);
        //System.out.println("------");
        //System.out.println(markerDump);
        
        try {
            saveCSV(markerDump, String.format("/home/rahm-lab/Desktop/MarkerTrackingGAStudien/csvdata/experiment-4/marker-dump-9-markers-all-cams.csv", isoDateAndTime()));
        } catch (IOException ex) {
            System.getLogger(ViconAPIDataExport.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
        }

        try {
            saveCSV(dump, String.format("/home/rahm-lab/Desktop/MarkerTrackingGAStudien/csvdata/experiment-4/centroid-dump-9-markers-all-cams.csv", isoDateAndTime()));
        } catch (IOException ex) {
            System.getLogger(ViconAPIDataExport.class.getName()).log(System.Logger.Level.ERROR, (String) null, ex);
        }
    }

    public static String csvCentroidDump(DataStreamClient client) {

        StringBuilder sb = new StringBuilder();

        sb.append(Camera.getCsvHeader()).append("\n");
        for (Camera camera : getCameras(client)) {
            sb.append(camera.toCsv());
        }

        return sb.toString();
    }

    public static String csvTrackingObjectDump(DataStreamClient client, ArrayList<String> objectNames) {
        
        StringBuilder sb = new StringBuilder();
        
        sb.append(TrackingObject.getCsvHeader()).append("\n");
        
        for (String objectName : objectNames) {
            TrackingObject trackingObject = getTrackingObject(client, objectName);
            sb.append(trackingObject.toCsv());
        }

        return sb.toString();
    }

    public static String isoDateAndTime() {
        // Get the current date
        LocalDateTime currentDate = LocalDateTime.now();

        // Format the date in ISO 8601 format
        String isoDate = currentDate.format(DateTimeFormatter.ISO_DATE_TIME);

        // Print the ISO formatted date
        return isoDate;
    }

    public static void saveCSV(String data, String path)
            throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(path));
        writer.write(data);
        writer.close();
    }

    public static ArrayList<ExportCentroid> getCentroids(DataStreamClient client, String cameraName) {
        ArrayList<ExportCentroid> centroids = new ArrayList<ExportCentroid>();

        long centroidCount = client.getCentroidCount(cameraName);

        for (long centroidIndex = 0; centroidIndex <= centroidCount - 1; centroidIndex++) {
            centroids.add(
                    new ExportCentroid(
                            client.getCentroidPosition(cameraName, centroidIndex).getPosition(),
                            client.getCentroidPosition(cameraName, centroidIndex).getRadius(),
                            client.getCentroidWeight(cameraName, centroidIndex),
                            centroidIndex
                    )
            );
        }

        return centroids;
    }

    public static ArrayList<Camera> getCameras(DataStreamClient client) {
        ArrayList<Camera> cameras = new ArrayList();

        for (long cameraIndex = 0; cameraIndex <= client.getCameraCount() - 1; cameraIndex++) {

            String cameraName = client.getCameraName(cameraIndex);

            cameras.add(new Camera(
                    cameraIndex,
                    cameraName,
                    client.getCameraUserId(cameraName),
                    client.getCameraDisplayName(cameraName),
                    getCentroids(client, cameraName)
            )
            );
        }

        return cameras;
    }

    public static TrackingObject getTrackingObject(DataStreamClient client, String objectName) {
        ArrayList<Marker> markers = new ArrayList();

        for (long i = 0; i <= client.getMarkerCount(objectName) - 1; i++) {
            String markerName = client.getMarkerName(objectName, i);

            double[] markerPos = client.getMarkerGlobalTranslation(objectName, markerName);

            markers.add(new Marker(
                    markerName,
                    markerPos
            ));
        }

        return new TrackingObject(objectName, markers);
    }
    
    public static ArrayList<Marker> getUnlabeledMarkers(DataStreamClient client){
        ArrayList<Marker> markers = new ArrayList();
        
        
        
        System.out.println("DEBUG: " + client.getUnlabeledMarkerCount());

        for (long i = 0; i <= client.getUnlabeledMarkerCount() - 1; i++) {
            double[] markerPos = client.getUnlabeledMarkerGlobalTranslation(i);
            markers.add(new Marker("unlabeled", markerPos));
        }
        
        return markers;
    }
    
    public static String csvUnlabeledMarkerDump(DataStreamClient client) {

        ArrayList<Marker> markers = getUnlabeledMarkers(client);
        
        StringBuilder sb = new StringBuilder();
        sb.append(Marker.getCsvHeader()).append("\n");
        
        for(Marker m:markers){
            sb.append(m.toCsv()).append("\n");
        }
        
        return sb.toString();
    }
}
