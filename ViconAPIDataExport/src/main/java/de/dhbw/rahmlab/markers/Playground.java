/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package de.dhbw.rahmlab.markers;

import de.dhbw.rahmlab.vicon.datastream.api.DataStreamClient;
import de.dhbw.rahmlab.vicon.datastream.api.Version;
import de.dhbw.rahmlab.vicon.datastream.impl.StreamMode_Enum;

/**
 *
 * @author rahm-lab
 */
public class Playground {

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
        System.out.println("Ulabeled Marker Data Enabled: " + client.isUnlabeledMarkerDataEnabled());

        client.getFrame();
        //for some reason I need a second getFrame() here, otherwise camera count is 0
        client.getFrame();

        System.out.println("Got: " + client.getCameraCount() + " cameras.");
        System.out.println("Got: " + client.getSubjectCount() + " Subjects");
        client.getFrame();
        client.getFrame();
        client.getFrame();
        System.out.println("Got: " + client.getLabeledMarkerCount() + " labeled Markers");
        System.out.println("Got: " + client.getUnlabeledMarkerCount() + " unlabeled markers.");
       
        
        for(long i = 0; i < client.getUnlabeledMarkerCount(); i++){
            double[] markerPos = client.getUnlabeledMarkerGlobalTranslation(i);
            System.out.println("(" + markerPos[0] + ", " + markerPos[1] + ", " + markerPos[2] + ")");
        }
    }
}
