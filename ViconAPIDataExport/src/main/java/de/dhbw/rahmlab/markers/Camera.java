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
public class Camera {
    private long index;
    private String name;
    private long userId;
    private String displayName;
    private ArrayList<ExportCentroid> centroids;

    public Camera(long index, String name, long userId, String displayName, ArrayList centroids) {
        this.index = index;
        this.name = name;
        this.userId = userId;
        this.displayName = displayName;
        this.centroids = centroids;
    }

    public long getIndex() {
        return index;
    }

    public String getName() {
        return name;
    }

    public long getUserId() {
        return userId;
    }

    public String getDisplayName() {
        return displayName;
    }

    public ArrayList<ExportCentroid> getCentroids() {
        return centroids;
    }
    
    public static String getCsvHeader() {
        return "cameraIndex,cameraName,cameraUserId,cameraDisplayName," + ExportCentroid.getCsvHeader();
    }
    
        public String toCsv() {
        StringBuilder sb = new StringBuilder();
        
        if (centroids.isEmpty()) {
            sb.append(getIndex()).append(",")
            .append(getName()).append(",")
            .append(getUserId()).append(",")
            .append(getDisplayName()).append(",")
            .append(",,,,").append("\n"); // No centroids, so empty fields for centroid info
        } else {
            for (ExportCentroid centroid : centroids) {
                        sb.append(getIndex()).append(",")
                        .append(getName()).append(",")
                        .append(getUserId()).append(",")
                        .append(getDisplayName()).append(",")
                        .append(centroid.toCsv()).append("\n");
            }
        }

        return sb.toString();
    }
}
