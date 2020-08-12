package com.databricks.labs.automl.exploration.analysis.trees.scripts

import com.databricks.labs.automl.exploration.analysis.common.encoders.Converters
import com.databricks.labs.automl.exploration.analysis.common.structures.{
  FeatureImportanceData,
  ModelData
}

private[analysis] object HTMLGenerators {

  def buildFeatureImportancesTable(
    importanceData: FeatureImportanceData
  ): String = {

    s"""
    <style>
    #fi {
      font-family: Arial, Helvetica, sans-serif;
      border-collapse: collapse;
      width: 100%;
    }
    #fi caption{
    caption-side: bottom;
    }
    #fi td {
      border: 2px solid #ddd;
      padding: 6px;
      text-align: left;
    }

    #fi th {
      padding-top: 12px;
      padding-bottom: 12px;
      text-align: center;
      background-color: #00b379;
    }
    #fi tr:nth-child(odd){background-color: #f2f2f2;}
    #fi tr:hover{background-color: #00b37984;}
    </style>
    <div style="overflow-x:auto;">
    ${Converters
      .mapFieldsToImportances(importanceData)
      .mkString(
        "<table id=fi><caption>Feature Importances</caption><tr><th>Feature</th><th>Importance</th></tr>",
        "",
        "</table>"
      )}
    </div>
    """

  }

  def buildFeatureImportanceChart(
    importanceData: FeatureImportanceData
  ): String = {

    s"""
       |<!DOCTYPE html>
       |<meta charset="utf-8">
       |<style>
       |
       |body {
       |  font: 12px sans-serif;
       |}
       |.axis path,
       |.axis line {
       |  fill: none;
       |  stroke: #D4D8DA;
       |  stroke-width: 1px;
       |  shape-rendering: crispEdges;
       |}
       |.bar {
       |  fill: lightsteelblue;
       |}
       |.bar:hover {
       |  fill: steelblue ;
       |}
       |.toolTip {
       |  position: absolute;
       |  display: none;
       |  min-width: 80px;
       |  height: auto;
       |  background: #D5D7E6;
       |  border: 1px solid #6F257F;
       |  padding: 14px;
       |  border-radius: 10px;
       |  pointer-events: none;
       |  text-align: center;
       |}
       |.svg-container {
       |  display: inline-block;
       |  position: relative;
       |  width: 60%;
       |  padding-bottom: 60%; /* aspect ratio */
       |  vertical-align: top;
       |  overflow: hidden;
       |}
       |.svg-content-responsive {
       |  display: inline-block;
       |  position: absolute;
       |  top: 10px;
       |  left: 0;
       |}
       |</style>
       |<body>
       |<script src="https://d3js.org/d3.v4.min.js"></script>
       |<div id="chartId"></div>
       |<script>
       |
       |var data = ${Converters.extractFieldImportancesAsJSON(importanceData)};
       |
       |var margin = {top: 20, right: 20, bottom: 200, left: 40},
       |    width = 500 - margin.left - margin.right,
       |    height = 500 - margin.top - margin.bottom;
       |
       |var x = d3.scaleBand()
       |          .range([0, width])
       |          .padding(0.1);
       |var y = d3.scaleLinear()
       |          .range([height, 0]);
       |
       |var svg = d3.select("div#chartId")
       |    .append("div")
       |    .classed("svg-container", true)
       |    .append("svg")
       |    .attr("preserveAspectRatio", "xMinYmin meet")
       |    .attr("viewBox", "0 0 500 500")
       |    .classed("svg-content-responsive", true)
       |    .append("g")
       |    .attr("transform", 
       |          "translate(" + margin.left + "," + margin.top + ")");
       |
       |var tooltip = d3.select("body").append("div").attr("class", "toolTip");
       |
       |var g = svg.append("g")
       |    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
       |
       |x.domain(data.map(function(d) { return d.feature; }));
       |y.domain([0, d3.max(data, function(d) { return d.importance; })]);
       |
       |g.append("g")
       |        .attr("class", "axis axis--x")
       |        .attr("transform", "translate(0," + height + ")")
       |        .call(d3.axisBottom(x))
       |        .selectAll("text")
       |        .style("text-anchor", "end")
       |        .attr("dx", "-0.8em")
       |        .attr("dy", "0.15em")
       |        .attr("transform", "rotate(-45)")
       |        .attr("font-size", "12px");
       |
       |g.append("g")
       |  	.attr("class", "axis axis--y")
       |  	.call(d3.axisLeft(y).ticks(10).tickSizeInner([-width]))
       |    .append("text")
       |    .attr("transform", "rotate(-90)")
       |    .attr("y", 6)
       |    .attr("dy", "-2.71em")
       |    .attr("text-anchor", "end")
       |    .attr("fill", "#5D6971")
       |    .attr("font-size", "14px")
       |    .text("Feature Importances");
       |
       |g.selectAll(".bar")
       |  .data(data)
       |  .enter().append("rect")
       |  .attr("class", "bar")
       |  .attr("x", function(d) { return x(d.feature); })
       |  .attr("width", x.bandwidth())
       |  .attr("y", function(d) { return y(d.importance); })
       |  .attr("height", function(d) { return height - y(d.importance); })
       |   .on("mousemove", function(d){
       |            tooltip
       |              .style("left", d3.event.pageX - 50 + "px")
       |              .style("top", d3.event.pageY - 70 + "px")
       |              .style("display", "inline-block")
       |              .html("Feature: " + (d.feature) + "<br>" + "Importance: " + (d.importance));
       |        })
       |    .on("mouseout", function(d){ tooltip.style("display", "none");});
       |   
       |
       |</script>
       |</body>
       |""".stripMargin

  }

  def createD3TreeVisualization(treeData: String,
                                mode: String = "static",
                                modelData: ModelData): String = {

    mode match {
      case "static" =>
        s"""
           |<!DOCTYPE html>
           |<html lang="en">
           |<head>
           |<meta charset="utf-8">
           |<style>
           |body{
           |font: 12px sans-serif;
           |}
           |.node {
           |      cursor: pointer;
           |  }
           |  .node circle {
           |    fill: #fff;
           |    stroke: steelblue;
           |    stroke-width: 3px;
           |  }
           |
           |  .node text {
           |    font: 10px sans-serif;
           |  }
           |
           |  .link {
           |    fill: none;
           |    stroke: #ccc;
           |    stroke-width: 2px;
           |  }
           |.tooltip{
           |    position: absolute;
           |    display: none;
           |    min-width: 80px;
           |    height: auto;
           |    background: #D5D7E6;
           |    border: 1px solid #6F257F;
           |    padding: 4px;
           |    border-radius: 10px;
           |    pointer-events: none;
           |    text-align: left;
           |}
           |.svg-container{
           |    display: inline-block;
           |    position: absolute;
           |    height: 200px;
           |    width: 1800px;
           |    padding-bottom: 95%;
           |    vertical-align: top;
           |    overflow-x: scroll;
           |    overflow-y: scroll;
           |}
           |.svg-content-responsive{
           |    display: inline-block;
           |    position: relative;
           |    top: 10px;
           |    left: 0px;
           |}
           |
           |</style>
           |</head>
           |<body>
           |
           |<!-- load the d3.js library -->	
           |<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min.js"></script>
           |<div id="chartId"></div>
           |<script>
           |
           |var treeData = $treeData
           |
           |// ************** Generate the tree diagram	 *****************
           |
           |var colors = ["#ff0000", "#ff1100", "#ff2200", "#ff3300", "#ff4400", "#ff5500", "#ff6600", "#ff7700", "#ff8800", "#ff9900", "#ffaa00", "#ffbb00", "#ffcc00", "#ffdd00", "#ffee00", "#ffff00", "#eeff00", "#ddff00", "#ccff00", "#bbff00", "#aaff00", "#99ff00", "#88ff00", "#77ff00", "#66ff00", "#55ff00", "#44ff00", "#33ff00", "#22ff00", "#11ff00", "#00ff00"]
           |
           |var colorScale = d3.scale.quantize()
           |  .domain([0, colors.length - 1, 1])
           |  .range(colors);
           |  
           |var margin = {top: 10, right: 100, bottom: 20, left: 100},
           |	width = 1000 - margin.right - margin.left,
           |	height = 1200 - margin.top - margin.bottom;
           |	
           |var i = 0,
           |	duration = 500,
           |	root;
           |
           |var tree = d3.layout.tree()
           |	.size([height, width]);
           |
           |var diagonal = d3.svg.diagonal()
           |	.projection(function(d) { return [d.y, d.x]; });
           |
           |var svg = d3.select("div#chartId")
           |    .append("div")
           |    .classed("svg-container", true)
           |    .append("svg")
           |    .classed("svg-content-responsive", true)
           |    .attr("preserveAspectRatio", "xMinYmin meet")
           |    .attr("width", "5000px")
           |    .attr("height", "2000px")
           |    .append("g")
           |    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
           |
           |root = treeData;
           |root.x0 = height / 2;
           |root.y0 = 0;
           |  
           |update(root);
           |
           |d3.select(self.frameElement).style("height", "1000px");
           |
           |function update(source) {
           |
           |  // Compute the new tree layout.
           |  var nodes = tree.nodes(root).reverse(),
           |	  links = tree.links(nodes);
           |
           |  // Normalize for fixed-depth.
           |  nodes.forEach(function(d) { d.y = d.depth * 260; });
           |
           |  var tooltip = d3.select("#chartId").append("div").attr("class", "tooltip");
           |  tooltip.append("div").attr("class", "feature");
           |  tooltip.append("div").attr("class", "informationGain");
           |  
           |
           |  // Update the nodes…
           |  var node = svg.selectAll("g.node")
           |	  .data(nodes, function(d) { return d.id || (d.id = ++i); });
           |
           |  // Enter any new nodes at the parent's previous position.
           |  var nodeEnter = node.enter().append("g")
           |	  .attr("class", "node")
           |	  .attr("transform", function(d) { return "translate(" + source.y0 + "," + source.x0 + ")"; })
           |	  
           |      .on("mousemove", function(d){
           |        
           |        tooltip.select(".feature")
           |            .html("<strong style = 'color: black'>Feature Name: </string style><strong style = 'color: red'>" + (d.name) + "</strong style>");
           |        tooltip.select(".informationGain")
           |            .html("<strong style = 'color:black'>Information Gain: </strong style><strong style = 'color:green'>" + (d.informationGain) + "</strong style>");
           |          
           |          tooltip
           |            .style("left", d3.event.pageX - 20 + "px")
           |            .style("top", d3.event.pageY - 40 + "px")
           |            .style("display", "block");
           |            
           |      })
           |      .on("mouseout", function(d){tooltip.style("display", "none")})
           |      .on("click", click)  
           |      ;
           |
           |
           |  nodeEnter.append("circle")
           |	  .attr("r", 1e-6)
           |      .style("fill", function(d){ return d._children ? "green" : "white"})
           |
           |  // Feature Name Text
           |  nodeEnter.append("text")
           |    .attr("dx", function(d){ return d._children || d.children ? 0: 15})
           |    .attr("dy", function(d){return d._children || d.children ? "2.5em": "0em"})
           |    .attr("text-anchor", function(d) {return d._children || d.children ? "middle": "start"})
           |    .attr("fill", "red")
           |    .text(function(d) { if(d.children || d._children) {return "Feature: " + d.name}; })
           |    .attr("font-family","serif")
           |
           |  // Prediction Text
           |  nodeEnter.append("text")
           |    .attr("dx", function(d){ return d._children || d.children ? 0: 15})
           |    .attr("dy", function(d){return d._children || d.children ? "3.5em": "0.35em"})
           |    .attr("text-anchor", function(d) {return d._children || d.children ? "middle": "start"})
           |    .text(function(d){
           |      if(typeof d.prediction !== "undefined"){
           |          return "Prediction: " + d.prediction
           |      } else {
           |          return "Prediction N/A"
           |      }
           |  })
           |    
           |  // Information Gain Text
           |  nodeEnter.append("text")
           |    .attr("dx", 0)
           |  .attr("dy", "4.5em")
           |    .attr("text-anchor", "middle")
           |  .text(function(d){if(typeof d.informationGain !== 'undefined') return "Information Gain: " + d.informationGain;})
           |  
           |  // Split Threshold Text  
           |  nodeEnter.append("text")
           |    .attr("dx", 0)
           |  .attr("dy", "5.5em")
           |    .attr("text-anchor", "middle")
           |  .text(function(d){
           |      if(typeof d.continuousSplitThreshold !== 'undefined')  { return "Split Threshold: " + d.continuousSplitThreshold } else if(d.children || d._children){ return "Split Criteria: Left -> [" + d.leftNodeCategories + "] Right-> [" + d.rightNodeCategories +"]"}
           |      })
           |  
           |  // Split Type Text
           |  nodeEnter.append("text")
           |    .attr("dx", 0)
           |  .attr("dy", "6.5em") 
           |    .attr("text-anchor", "middle")
           |  .text(function(d){if(typeof d.splitType !== 'undefined') return "Split Type: " + d.splitType;})
           | 
           |
           |  // Transition nodes to their new position.
           |  var nodeUpdate = node.transition()
           |	  .duration(duration)
           |	  .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });
           |
           |  nodeUpdate.select("circle")
           |	  .attr("r", 12)
           |      .style("fill", function(d){return colorScale(d.informationGain)})
           |      .style("stroke", function(d){return d._children || d.children ? "black" : "lightgrey"})
           |
           |  nodeUpdate.select("text")
           |	  .style("fill-opacity", 1);
           |
           |  // Transition exiting nodes to the parent's new position.
           |  var nodeExit = node.exit().transition()
           |	  .duration(duration)
           |	  .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
           |	  .remove();
           |
           |  nodeExit.select("circle")
           |	  .attr("r", 1e-6);
           |
           |  nodeExit.select("text")
           |	  .style("fill-opacity", 1e-6);
           |
           |  // Update the links…
           |  var link = svg.selectAll("path.link")
           |	  .data(links, function(d) { return d.target.id; });
           |
           |  // Enter any new links at the parent's previous position.
           |  link.enter().insert("path", "g")
           |	  .attr("class", "link")
           |	  .attr("d", function(d) {
           |		var o = {x: source.x0, y: source.y0};
           |		return diagonal({source: o, target: o});
           |	  });
           |
           |  // Transition links to their new position.
           |  link.transition()
           |	  .duration(duration)
           |	  .attr("d", diagonal);
           |
           |  // Transition exiting nodes to the parent's new position.
           |  link.exit().transition()
           |	  .duration(duration)
           |	  .attr("d", function(d) {
           |		var o = {x: source.x, y: source.y};
           |		return diagonal({source: o, target: o});
           |	  })
           |	  .remove();
           |
           |  // Stash the old positions for transition.
           |  nodes.forEach(function(d) {
           |	d.x0 = d.x;
           |	d.y0 = d.y;
           |  });
           |}
           |
           |// Toggle children on click.
           |function click(d) {
           |  if (d.children) {
           |	d._children = d.children;
           |	d.children = null;
           |  } else {
           |	d.children = d._children;
           |	d._children = null;
           |  }
           |  update(d);
           |}
           |
           |</script>
           |	
           |  </body>
           |</html>
           |""".stripMargin
      case "dynamic" =>
        s"""
           |<!DOCTYPE html>
           |<html lang="en">
           |<head>
           |<meta charset="utf-8">
           |<style>
           |body{
           |font: 12px sans-serif;
           |}
           |.node {
           |      cursor: pointer;
           |  }
           |  .node circle {
           |    fill: #fff;
           |    stroke: steelblue;
           |    stroke-width: 3px;
           |  }
           |
           |  .node text {
           |    font: 12px sans-serif;
           |    font-weight: 700;
           |  }
           |
           |  .link {
           |    fill: none;
           |    stroke: #ccc;
           |    stroke-width: 2px;
           |  }
           |  .link text {
           |      font: 10px sans-serif;
           |      fill: lightblue;
           |  }
           |.tooltip{
           |    position: absolute;
           |    display: none;
           |    min-width: 80px;
           |    height: auto;
           |    background: #D5D7E6;
           |    border: 1px solid #6F257F;
           |    padding: 4px;
           |    border-radius: 10px;
           |    pointer-events: none;
           |    text-align: left;
           |}
           |.svg-container{
           |    display: inline-block;
           |    position: absolute;
           |    height: 200px;
           |    width: 1800px;
           |    padding-bottom: 95%;
           |    vertical-align: top;
           |    overflow-x: scroll;
           |    overflow-y: scroll;
           |}
           |.svg-content-responsive{
           |    display: inline-block;
           |    position: relative;
           |    top: 10px;
           |    left: 0px;
           |}
           |
           |</style>
           |</head>
           |<body>
           |
           |<!-- load the d3.js library -->	
           |<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min.js"></script>
           |<div id="chartId"></div>
           |<script>
           |
           |var treeData = $treeData
           |
          var colors = ["#ff0000", "#ff1100", "#ff2200", "#ff3300", "#ff4400", "#ff5500", "#ff6600", "#ff7700", "#ff8800", "#ff9900", "#ffaa00", "#ffbb00", "#ffcc00", "#ffdd00", "#ffee00", "#ffff00", "#eeff00", "#ddff00", "#ccff00", "#bbff00", "#aaff00", "#99ff00", "#88ff00", "#77ff00", "#66ff00", "#55ff00", "#44ff00", "#33ff00", "#22ff00", "#11ff00", "#00ff00"]
           |
           |var colorScale = d3.scale.quantize()
           |  .domain([0, colors.length - 1, 1])
           |  .range(colors);
           |
           |
           |// ************** Generate the tree diagram	 *****************
           |   
           |    
           |var margin = {top: 10, right: 100, bottom: 20, left: 100},
           |	width = 1000 - margin.right - margin.left,
           |	height = 1200 - margin.top - margin.bottom;
           |	
           |var i = 0,
           |	duration = 500,
           |	root;
           |
           |var tree = d3.layout.tree()
           |	.size([height, width]);
           |
           |var diagonal = d3.svg.diagonal()
           |	.projection(function(d) { return [d.y, d.x]; });
           |
           |var svg = d3.select("div#chartId")
           |    .append("div")
           |    .classed("svg-container", true)
           |    .append("svg")
           |    .classed("svg-content-responsive", true)
           |    .attr("preserveAspectRatio", "xMinYmin meet")
           |    .attr("width", "5000px")
           |    .attr("height", "2000px")
           |    .append("g")
           |    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
           |
           |root = treeData;
           |root.x0 = height / 2;
           |root.y0 = 0;
           |  
           |function collapse(d) {
           |  if (d.children) {
           |    d._children = d.children;
           |    d._children.forEach(collapse);
           |    d.children = null;
           |  }
           |}
           |
           |update(root);
           |
           |// d3.select(self.frameElement).style("height", "600px");
           |
           |function update(source) {
           |
           |  // Compute the new tree layout.
           |  var nodes = tree.nodes(root).reverse(),
           |    links = tree.links(nodes);
           |
           |  // Normalize for fixed-depth.
           |  nodes.forEach(function(d) {
           |    d.y = d.depth * 260;
           |  });
           |
           |  // Update the nodes…
           |  var node = svg.selectAll("g.node")
           |    .data(nodes, function(d) {
           |      return d.id || (d.id = ++i);
           |    });
           |
           |var textColor = 'white'
           |var boxColor = 'black'
           |
           |  // Enter any new nodes at the parent's previous position.
           | var nodeEnter = node.enter().append("g")
           |    .attr("class", "node")
           |    .attr("transform", function(d) {
           |      return "translate(" + source.y0 + "," + source.x0 + ")";
           |    })
           |    
           |  nodeEnter.append("circle")
           |    .attr("r", 1e-6)
           |    .style("fill", function(d) {
           |      return d._children ? "green" : "white";
           |    });
           |
           |  nodeEnter.append("text")
           |    .attr("x", function(d) {
           |      return d.children || d._children ? -16 : 16;
           |    })
           |    .attr("dy", ".35em")
           |    .attr("text-anchor", function(d) {
           |      return d.children || d._children ? "end" : "start";
           |    })
           |    .text(function(d) {
           |      return d.name;
           |    })
           |
           |  // Transition nodes to their new position.
           |  var nodeUpdate = node.transition()
           |    .duration(duration)
           |    .attr("transform", function(d) {
           |      return "translate(" + d.y + "," + d.x + ")";
           |    });
           |
           |  nodeUpdate.select("circle")
           |	  .attr("r", 12)
           |      .style("fill", function(d){return colorScale(d.informationGain)})
           |      .style("stroke", function(d){return d._children || d.children ? "black" : "lightgrey"})
           |
           |
           |  nodeUpdate.select("text")
           |    .style("fill-opacity", 1);
           |
           |  // Transition exiting nodes to the parent's new position.
           |  var nodeExit = node.exit().transition()
           |    .duration(duration)
           |    .attr("transform", function(d) {
           |      return "translate(" + source.y + "," + source.x + ")";
           |    })
           |    .remove();
           |
           |  nodeExit.select("circle")
           |    .attr("r", 1e-6);
           |
           |  nodeExit.select("text")
           |    .style("fill-opacity", 1e-6);
           |
           |  // Update the links…
           |  var link = svg.selectAll("path.link")
           |    .data(links, function(d) {
           |      return d.target.id;
           |    })
           |
           |  // Enter any new links at the parent's previous position.
           |  link.enter().insert("path", "g")
           |    .attr("class", "link")
           |    .attr("d", function(d) {
           |      var o = {
           |        x: source.x0,
           |        y: source.y0
           |      };
           |      return diagonal({
           |        source: o,
           |        target: o
           |      });
           |    });
           |
           |  // Transition links to their new position.
           |  link.transition()
           |    .duration(duration)
           |    .attr("d", diagonal);
           |
           |  // Transition exiting nodes to the parent's new position.
           |  link.exit().transition()
           |    .duration(duration)
           |    .attr("d", function(d) {
           |      var o = {
           |        x: source.x,
           |        y: source.y
           |      };
           |      return diagonal({
           |        source: o,
           |        target: o
           |      });
           |    })
           |    .remove();
           |
           |  // Stash the old positions for transition.
           |  nodes.forEach(function(d) {
           |    d.x0 = d.x;
           |    d.y0 = d.y;
           |  });
           |  
           |  var headX = '25em'
           |  var nodeX = '25em'
           |  var leafX = '-4em'
           |  var headRectX = '2em'
           |  var nodeRectX = '2em'
           |  var leafRectX = '-20em'
           |  
           |  nodeEnter.on("click", click)
           |    .on("mouseover", function(d) {
           |      var g = d3.select(this); // The node
           |      // The class is used to remove the additional text later
           |      var rect = g.append('rect')
           |        .classed('box', true)
           |        .attr('x', function(d){
           |            if(d.name == "LeafNode"){return leafRectX} else if(!(d.parent)){return headRectX} else { return nodeRectX}})
           |        .attr('y', function(d){if(d.name == "LeafNode"){return "-1.5em"} else {return '-3.5em'}})
           |        .attr('width', function(d){if(d.name == "LeafNode"){return "18em"} else {return '24.5em'}})
           |        .attr('height', function(d){if(d.name == "LeafNode"){return "3em"} else {return "7em"}})
           |        .attr("rx", 10)
           |        .attr("ry", 10)
           |        .style('fill', "grey")
           |        .style('opacity', 1)
           |        .attr('stroke-width', 2)
           |        .attr('stroke', 'darkgrey');
           |        
           |        d3.select("box")
           |        .append("defs").append("filter")
           |        .attr("id", "blur")
           |        .append("feGaussianBlur")
           |        .attr("stdDeviation", 5);
           |        
           |        
           |      var info = g.append('text')
           |        .classed('info', true)
           |        .attr('dx', function(d){
           |            if(d.name == "LeafNode"){return leafX} else if(!(d.parent)){return headX} else { return nodeX}})
           |        .attr('dy', function(d){if(d.name == "LeafNode"){return "0.5em"}else{return"2.5em"}})
           |        .attr('text-anchor', 'end')
           |        .style('opacity', 1)
           |        .attr('fill', "lightgreen")
           |        .text(function(d){
           |          if(typeof d.prediction !== "undefined"){
           |              return "Prediction: " + d.prediction;
           |          } else {
           |              return "Prediction N/A";
           |          }
           |        })
           |      var info2 = g.append('text')
           |      .classed('info2', true)
           |      .attr('dx', function(d){
           |            if(d.name == "LeafNode"){return leafX} else if(!(d.parent)){return headX} else { return nodeX}})
           |      .attr('dy', "1.5em")
           |      .attr("text-anchor", 'end')
           |      .attr('fill', textColor)
           |      .text(function(d){
           |          if(typeof d.informationGain !== 'undefined'){
           |          return "Information Gain: " + d.informationGain;
           |          }})
           |      var info3 = g.append('text')
           |        .classed('info3', true)
           |        .attr('dx', function(d){
           |            if(d.name == "LeafNode"){return leafX} else if(!(d.parent)){return headX} else { return nodeX}})
           |        .attr('dy', '0.5em')
           |        .attr('text-anchor', 'end')
           |        .attr('fill', textColor)
           |        .text(function(d){
           |            if(typeof d.continuousSplitThreshold !== 'undefined')  { return "Split Threshold: " + d.continuousSplitThreshold } else if(d.children || d._children){ return "Split Criteria: Left -> " + d.leftNodeCategories + "  " + "Right-> " + d.rightNodeCategories}
           |        })
           |      var info4 = g.append('text')
           |        .classed('info4', true)
           |        .attr('dx', function(d){
           |            if(d.name == "LeafNode"){return leafX} else if(!(d.parent)){return headX} else { return nodeX}})
           |        .attr('dy', '-0.5em')
           |        .attr('text-anchor', 'end')
           |        .attr('fill', textColor)
           |        .text(function(d){if(typeof d.splitType !== 'undefined') return "Split Type: " + d.splitType;})
           |      var info5 = g.append('text')
           |        .classed('info5', true)
           |        .attr('dx', function(d){
           |            if(d.name == "LeafNode"){return leafX} else if(!(d.parent)){return headX} else { return nodeX}})
           |        .attr('dy', '-1.5em')
           |        .attr('text-anchor', 'end')
           |        .attr('fill', "springgreen")
           |        .attr('font-weight', 500)
           |        .text(function(d){if(d.children || d._children) {return "Feature: " + d.name}})
           |    })
           |    
           |    
           |    
           |    .on("mouseout", function() {
           |      // Remove the info text on mouse out.
           |      d3.select(this).select('text.info').remove();
           |      d3.select(this).select('text.info2').remove();
           |      d3.select(this).select('text.info3').remove();
           |      d3.select(this).select('text.info4').remove();
           |      d3.select(this).select('text.info5').remove();
           |      d3.select(this).select('rect.box').remove();
           |    });
           |  
           |}
           |
           |// Toggle children on click.
           |function click(d) {
           |  if (d.children) {
           |    d._children = d.children;
           |    d.children = null;
           |  } else {
           |    d.children = d._children;
           |    d._children = null;
           |  }
           |  update(d);
           |}
           |
           |
           |</script>
           |	
           |  </body>
           |</html>
       """.stripMargin
      case "lightweight" =>
        s"""
           |<!DOCTYPE html>
           |<html lang="en">
           |<head>
           |<meta charset="utf-8">
           |<style>
           |body{
           |font: 12px sans-serif;
           |}
           |.node {
           |      cursor: pointer;
           |  }
           |  .node circle {
           |    fill: #fff;
           |    stroke: steelblue;
           |    stroke-width: 3px;
           |  }
           |
           |  .node text {
           |    font: 12px sans-serif;
           |    font-weight: 700;
           |  }
           |
           |  .link {
           |    fill: none;
           |    stroke: #ccc;
           |    stroke-width: 2px;
           |  }
           |  .link text {
           |      font: 10px sans-serif;
           |      fill: lightblue;
           |  }
           |.tooltip{
           |    position: absolute;
           |    display: none;
           |    min-width: 80px;
           |    height: auto;
           |    background: #D5D7E6;
           |    border: 1px solid #6F257F;
           |    padding: 4px;
           |    border-radius: 10px;
           |    pointer-events: none;
           |    text-align: left;
           |}
           |.svg-container{
           |    display: inline-block;
           |    position: absolute;
           |    height: 200px;
           |    width: 1800px;
           |    padding-bottom: 95%;
           |    vertical-align: top;
           |    overflow-x: scroll;
           |    overflow-y: scroll;
           |}
           |.svg-content-responsive{
           |    display: inline-block;
           |    position: relative;
           |    top: 10px;
           |    left: 0px;
           |}
           |
           |</style>
           |</head>
           |<body>
           |
           |<!-- load the d3.js library -->	
           |<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min.js"></script>
           |<div id="chartId"></div>
           |<script>
           |var treeData = ${treeData}
           |
           |// ************** Generate the tree diagram	 *****************
           |   
           |    
           |var margin = {top: 10, right: 100, bottom: 20, left: 100},
           |	width = 1000 - margin.right - margin.left,
           |	height = 1200 - margin.top - margin.bottom;
           |	
           |var i = 0,
           |	duration = 500,
           |	root;
           |
           |var tree = d3.layout.tree()
           |	.size([height, width]);
           |
           |var diagonal = d3.svg.diagonal()
           |	.projection(function(d) { return [d.y, d.x]; });
           |
           |var svg = d3.select("div#chartId")
           |    .append("div")
           |    .classed("svg-container", true)
           |    .append("svg")
           |    .classed("svg-content-responsive", true)
           |    .attr("preserveAspectRatio", "xMinYmin meet")
           |    .attr("width", "5000px")
           |    .attr("height", "2000px")
           |    .append("g")
           |    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
           |
           |root = treeData;
           |root.x0 = height / 2;
           |root.y0 = 0;
           |  
           |function collapse(d) {
           |  if (d.children) {
           |    d._children = d.children;
           |    d._children.forEach(collapse);
           |    d.children = null;
           |  }
           |}
           |
           |update(root);
           |
           |function update(source) {
           |
           |  // Compute the new tree layout.
           |  var nodes = tree.nodes(root).reverse(),
           |	  links = tree.links(nodes);
           |
           |  // Normalize for fixed-depth.
           |  nodes.forEach(function(d) { d.y = d.depth * 260; });
           |
           |  var tooltip = d3.select("#chartId").append("div").attr("class", "tooltip");
           |  tooltip.append("div").attr("class", "feature");
           |  tooltip.append("div").attr("class", "informationGain");
           |  tooltip.append("div").attr("class", "prediction");
           |  tooltip.append("div").attr("class", "splitThreshold");
           |  tooltip.append("div").attr("class", "splitType")
           |  
           |
           |  // Update the nodes…
           |  var node = svg.selectAll("g.node")
           |	  .data(nodes, function(d) { return d.id || (d.id = ++i); });
           |
           |  // Enter any new nodes at the parent's previous position.
           |  var nodeEnter = node.enter().append("g")
           |	  .attr("class", "node")
           |	  .attr("transform", function(d) { return "translate(" + source.y0 + "," + source.x0 + ")"; })
           |	  
           |      .on("mouseover", function(d){
           |        
           |        var threshold = function(d){
           |            if(typeof d.continuousSplitThreshold !== 'undefined')  { return d.continuousSplitThreshold } else if(d.children || d._children){ return "Left -> [" + d.leftNodeCategories + "] Right-> [" + d.rightNodeCategories +"]"}
           |        };
           |        
           |        tooltip.select(".feature")
           |            .html("<strong style = 'color: black'>Feature Name: </string style><strong style = 'color: red'>" + (d.name) + "</strong style>");
           |        tooltip.select(".prediction")
           |            .html("<strong style = 'color: black'>Prediction: </string style><strong style = 'color: green'>" + (d.prediction) + "</strong style>");
           |        tooltip.select(".informationGain")
           |            .html("<strong style = 'color:black'>Information Gain: </strong style><strong style = 'color: blue'>" + (d.informationGain) + "</strong style>");
           |        tooltip.select(".splitThreshold")
           |            .html("<strong style = 'color: black'>Split Threshold: </string style><strong style = 'color: black'>" + threshold(d) + "</strong style>");
           |        tooltip.select(".splitType")
           |            .html("<strong style = 'color: black'>Split Type: </string style><strong style = 'color: black'>" + (d.splitType) + "</strong style>");
           |            
           |          tooltip
           |            .style("left", d3.event.pageX + 40 + "px")
           |            .style("top", d3.event.pageY - 40 + "px")
           |            .style("display", "block");
           |            
           |      })
           |      .on("mouseout", function(d){tooltip.style("display", "none")})
           |      .on("click", click)  
           |      ;
           |
           |
           |  nodeEnter.append("circle")
           |	  .attr("r", 1e-6)
           |      .style("fill", function(d){ return d._children ? "green" : "white"})
           |
           |  // Feature Name Text
           |  nodeEnter.append("text")
           |    .attr("dx", function(d){ return d._children || d.children ? 0: 15})
           |    .attr("dy", function(d){return d._children || d.children ? "2.5em": "0em"})
           |    .attr("text-anchor", function(d) {return d._children || d.children ? "middle": "start"})
           |    .attr("fill", "black")
           |    .attr("font-weight", 700)
           |    .text(function(d) { if(d.children || d._children) {return d.name}; })
           |    .attr("font-family","serif")
           |
           |  // Transition nodes to their new position.
           |  var nodeUpdate = node.transition()
           |	  .duration(duration)
           |	  .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });
           |
           |  nodeUpdate.select("circle")
           |	  .attr("r", 12)
           |	  .style("fill", function(d){ if(d.children){return "lightblue"} else if (d._children){return "lightgreen"} else {return "#fff"}})
           |	  .style("stroke", function(d){return d.children  || d._children ? "blue" : "lightsteelblue";});
           |
           |
           |  nodeUpdate.select("text")
           |	  .style("fill-opacity", 1);
           |
           |  // Transition exiting nodes to the parent's new position.
           |  var nodeExit = node.exit().transition()
           |	  .duration(duration)
           |	  .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
           |	  .remove();
           |
           |  nodeExit.select("circle")
           |	  .attr("r", 1e-6);
           |
           |  nodeExit.select("text")
           |	  .style("fill-opacity", 1e-6);
           |
           |  // Update the links…
           |  var link = svg.selectAll("path.link")
           |	  .data(links, function(d) { return d.target.id; });
           |
           |  // Enter any new links at the parent's previous position.
           |  link.enter().insert("path", "g")
           |	  .attr("class", "link")
           |	  .attr("d", function(d) {
           |		var o = {x: source.x0, y: source.y0};
           |		return diagonal({source: o, target: o});
           |	  });
           |
           |  // Transition links to their new position.
           |  link.transition()
           |	  .duration(duration)
           |	  .attr("d", diagonal);
           |
           |  // Transition exiting nodes to the parent's new position.
           |  link.exit().transition()
           |	  .duration(duration)
           |	  .attr("d", function(d) {
           |		var o = {x: source.x, y: source.y};
           |		return diagonal({source: o, target: o});
           |	  })
           |	  .remove();
           |
           |  // Stash the old positions for transition.
           |  nodes.forEach(function(d) {
           |	d.x0 = d.x;
           |	d.y0 = d.y;
           |  });
           |}
           |
           |// Toggle children on click.
           |function click(d) {
           |  if (d.children) {
           |	d._children = d.children;
           |	d.children = null;
           |  } else {
           |	d.children = d._children;
           |	d._children = null;
           |  }
           |  update(d);
           |}
           |
           |</script>
           |	
           |  </body>
           |</html>
           |""".stripMargin
      case _ =>
        throw new UnsupportedOperationException(
          s"mode $mode is not supported.  Must be either 'static', 'lightweight', or 'dynamic'"
        )
    }
  }

}
