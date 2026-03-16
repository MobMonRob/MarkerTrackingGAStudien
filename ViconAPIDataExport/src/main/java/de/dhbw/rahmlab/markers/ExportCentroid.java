/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package de.dhbw.rahmlab.markers;

import de.dhbw.rahmlab.vicon.datastream.api.Centroid;

/**
 *
 * @author rahm-lab
 */
public class ExportCentroid extends Centroid {

    private final double weight;
    private final long index;

    public ExportCentroid(double[] position, double radius, double weight, long index) {
        super(position, radius);
        this.weight = weight;
        this.index = index;
    }

    public double getWeight() {
        return weight;
    }

    public long getIndex() {
        return index;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("ExportCentroid{");
        sb.append("positionX=").append(getPosition()[0]);
        sb.append(", positionY=").append(getPosition()[1]);
        sb.append(", radius=").append(getRadius());
        sb.append(", weight=").append(weight);
        sb.append(", index=").append(index);
        sb.append('}');
        return sb.toString();
    }

    public String toCsv() {
        StringBuilder sb = new StringBuilder();

        // Append position array elements
        for (int i = 0; i < getPosition().length; i++) {
            sb.append(getPosition()[i]);
            if (i < getPosition().length - 1) {
                sb.append(",");  // Add comma between elements
            }
        }

        sb.append(",").append(getRadius());
        sb.append(",").append(getWeight());
        sb.append(",").append(getIndex());

        return sb.toString();
    }
    
    public static String getCsvHeader() {
        return "positionX,positionY,radius,weight,centroidIndex";
    }
}
