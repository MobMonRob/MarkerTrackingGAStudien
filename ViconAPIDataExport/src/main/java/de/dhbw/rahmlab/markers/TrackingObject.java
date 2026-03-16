/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package de.dhbw.rahmlab.markers;

import java.util.ArrayList;

/**
 *
 * @author rahm-lab
 */
class TrackingObject {

    private final String name;
    private final ArrayList<Marker> markers;

    public TrackingObject(String name, ArrayList<Marker> markers) {
        this.name = name;
        this.markers = markers;
    }

    public String getName() {
        return name;
    }

    public int getMarkerCount() {
        return markers.size();
    }

    public ArrayList<Marker> getMarkers() {
        return markers;
    }

    public String toCsv() {
        StringBuilder sb = new StringBuilder();

        if (getMarkers().isEmpty()) {
            sb.append(getName()).append(",,,,");
        } else {
            for (Marker marker : getMarkers()) {
                sb.append(getName()).append(",");
                sb.append(marker.toCsv()).append("\n");
            }
        }
        
        return sb.toString();
    }

    public static String getCsvHeader() {
        return "objectName," + Marker.getCsvHeader();
    }
}
