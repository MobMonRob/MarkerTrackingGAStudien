/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package de.dhbw.rahmlab.markers;

/**
 *
 * @author rahm-lab
 */
public class Marker {

    private String name;
    private double x;
    private double y;
    private double z;

    public Marker(String name, double[] position) {
        this.name = name;
        this.x = position[0];
        this.y = position[1];
        this.z = position[2];
    }

    public String getName() {
        return name;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public double getZ() {
        return z;
    }

    public String toCsv() {
        StringBuilder sb = new StringBuilder();

        sb.append(getName()).append(",")
            .append(getX()).append(",")
            .append(getY()).append(",")
            .append(getZ());
        return sb.toString();
    }

    public static String getCsvHeader() {
        return "markerName,markerPosX,markerPosY,markerPosZ";
    }
}
