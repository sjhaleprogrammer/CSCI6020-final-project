var ClickableMap = {};

(function () {
  var version = "1.0.0";
  var classPrefix = "cmm-usa-";
  var creditLinkUrl = "https://www.clickablemapmaker.com";
  var stateCount = 0;
  var maxTableColumns = 5;
  var global = {
    getEleById: function name(id) {
      return document.getElementById(id);
    },
    getEleByQuery: function (query) {
      return document.querySelector(query);
    },
    stateIdToDomClass: function (stateId) {
      return `state-${stateId}`;
    },
  };

  this.version = version;

  function createBaseGlobalData() {
    return {
      version: version,
      width: "800",
      widthUnits: "px",
      fontSize: "12px",
      fontName: "Arial",
      fill: "#97e2bb",
      hoverFill: "#ffffff",
      disabledFill: "#c2c2c2",
      backgroundFill: "#ffffff",
      innerLabelColor: "#000000",
      outerLabelColor: "#000000",
      hoverLabelColor: "#D64933",
      borderType: null,
      borderStroke: "#49bc95",
      enableShadows: true,
      popLink: false,
      showStateTitleAndDescOnHover: true,
      showLinksList: false,
      globalLinkUrl: null,
      globalJsCallback: null,
      mapTitle: "",
      creditLink: "",
    };
  }

  function createBaseStatesData() {
    var statesData = {
      AL: { fullName: "Alabama" },
      AK: { fullName: "Alaska" },
      AZ: { fullName: "Arizona" },
      AR: { fullName: "Arkansas" },
      CA: { fullName: "California" },
      CO: { fullName: "Colorado" },
      CT: { fullName: "Connecticut" },
      DE: { fullName: "Delaware" },
      DC: { fullName: "District Of Columbia" },
      FL: { fullName: "Florida" },
      GA: { fullName: "Georgia" },
      HI: { fullName: "Hawaii" },
      ID: { fullName: "Idaho" },
      IL: { fullName: "Illinois" },
      IN: { fullName: "Indiana" },
      IA: { fullName: "Iowa" },
      KS: { fullName: "Kansas" },
      KY: { fullName: "Kentucky" },
      LA: { fullName: "Louisiana" },
      ME: { fullName: "Maine" },
      MD: { fullName: "Maryland" },
      MA: { fullName: "Massachusetts" },
      MI: { fullName: "Michigan" },
      MN: { fullName: "Minnesota" },
      MS: { fullName: "Mississippi" },
      MO: { fullName: "Missouri" },
      MT: { fullName: "Montana" },
      NE: { fullName: "Nebraska" },
      NV: { fullName: "Nevada" },
      NH: { fullName: "New Hampshire" },
      NJ: { fullName: "New Jersey" },
      NM: { fullName: "New Mexico" },
      NY: { fullName: "New York" },
      NC: { fullName: "North Carolina" },
      ND: { fullName: "North Dakota" },
      OH: { fullName: "Ohio" },
      OK: { fullName: "Oklahoma" },
      OR: { fullName: "Oregon" },
      PA: { fullName: "Pennsylvania" },
      RI: { fullName: "Rhode Island" },
      SC: { fullName: "South Carolina" },
      SD: { fullName: "South Dakota" },
      TN: { fullName: "Tennessee" },
      TX: { fullName: "Texas" },
      UT: { fullName: "Utah" },
      VT: { fullName: "Vermont" },
      VA: { fullName: "Virginia" },
      WA: { fullName: "Washington" },
      WV: { fullName: "West Virginia" },
      WI: { fullName: "Wisconsin" },
      WY: { fullName: "Wyoming" },
    };

    for (var stateId in statesData) {
      if (!statesData.hasOwnProperty(stateId)) {
        continue;
      }

      statesData[stateId].title = statesData[stateId].fullName;
      statesData[stateId].description = null;
      statesData[stateId].longDescription = null;
      statesData[stateId].linkUrl = null;
      statesData[stateId].isDisabled = false;
      statesData[stateId].isHovering = false;
      statesData[stateId].cssClass = null;
      statesData[stateId].overrideFill = null;
      statesData[stateId].overrideFillEnabled = false;
      statesData[stateId].overrideHoverFill = null;
      statesData[stateId].overrideHoverFillEnabled = false;
      statesData[stateId].overridePopLink = null;

      stateCount++;
    }

    return statesData;
  }

  function stateOn(stateId) {
    // Exit if the state is already in a hovering state
    if (this.statesData[stateId].isHovering) {
      return;
    }

    // Set the state to hovering
    this.statesData[stateId].isHovering = true;

    // Ensure $map is initialized
    if (!this.$map || !this.$map.id) {
      console.error("Error: $map is not properly initialized or missing an id");
      return;
    }

    // Retrieve the necessary DOM elements
    var stateClass = global.stateIdToDomClass(stateId);

    var $stateLink = global.getEleByQuery(`#${this.$map.id} .${stateClass}`);
    if (!$stateLink) {
      console.error(`Error: State link for class "${stateClass}" not found`);
    }

    var $statePath = global.getEleByQuery(
      `#${this.$map.id} .${stateClass} path`,
    );
    if (!$statePath) {
      console.error(`Error: State path for class "${stateClass}" not found`);
    }

    var $stateText = global.getEleByQuery(
      `#${this.$map.id} .${stateClass} text`,
    );
    if (!$stateText) {
      console.error(`Error: State text for class "${stateClass}" not found`);
    }

    // Use the elements as needed
    console.log({ $stateLink, $statePath, $stateText });

    // Apply hover styles based on state conditions
    if (this.statesData[stateId].isDisabled) {
      $statePath.style.fill = this.globalData.disabledFill;
      $stateLink.style.cursor = "default";
    } else if (
      this.statesData[stateId].overrideHoverFillEnabled &&
      this.statesData[stateId].overrideHoverFill != null
    ) {
      $statePath.style.fill = this.statesData[stateId].overrideHoverFill;
      $stateText.style.fill = this.globalData.hoverLabelColor;
      $stateLink.style.cursor = "pointer";
    } else {
      $statePath.style.fill = this.globalData.hoverFill;
      $stateText.style.fill = this.globalData.hoverLabelColor;
      $stateLink.style.cursor = "pointer";
    }

    // Show title and description on hover if enabled
    if (this.globalData.showStateTitleAndDescOnHover) {
      var $hoverStateInfo = global.getEleByQuery(
        `#${this.$map.id} .${classPrefix}hover-state-info`,
      );

      var titleText = this.statesData[stateId].title || "";
      var descText = this.statesData[stateId].description || "";
      var longDescText = this.statesData[stateId].longDescription || "";

      var titleSpan = document.createElement("span");
      var descSpan = document.createElement("span");

      // Set content for title and description
      titleSpan.textContent = titleText;
      descSpan.innerHTML = longDescText || descText;

      // Clear previous content and append new content
      $hoverStateInfo.innerHTML = ""; // Clear all children
      $hoverStateInfo.appendChild(titleSpan);
      $hoverStateInfo.appendChild(descSpan);

      // Display hover state info
      $hoverStateInfo.style.display = "block";
    }

    // Add shadow effects if enabled
    if (!this.statesData[stateId].isDisabled && this.globalData.enableShadows) {
      var statePathBlur = $statePath.cloneNode(true);
      statePathBlur.setAttribute("filter", `url(#${this.$map.id}-blur-filter)`);
      statePathBlur.setAttribute("class", `${classPrefix}state-shadow`);
      $stateLink.parentNode.appendChild(statePathBlur);
      $stateLink.parentNode.appendChild($stateLink);
    }
  }

  function stateOff(stateId) {
    this.statesData[stateId].isHovering = false;

    var $statePath = global.getEleByQuery(
      "#" + this.$map.id + " ." + global.stateIdToDomClass(stateId) + " path",
    );

    var $stateText = global.getEleByQuery(
      "#" + this.$map.id + " ." + global.stateIdToDomClass(stateId) + " text",
    );
    var isOuterLabel =
      $stateText.getAttribute("class") == classPrefix + "outer-label";

    if (this.globalData.showStateTitleAndDescOnHover) {
      var $hoverStateInfo = global.getEleByQuery(
        "#" + this.$map.id + " ." + classPrefix + "hover-state-info",
      );
      $hoverStateInfo.style.display = "none";
    }
    if (this.statesData[stateId].isDisabled) {
      $statePath.style.fill = this.globalData.disabledFill;
    } else if (
      this.statesData[stateId].overrideFillEnabled &&
      this.statesData[stateId].overrideFill != null
    ) {
      $statePath.style.fill = this.statesData[stateId].overrideFill;
      $stateText.style.fill = isOuterLabel
        ? this.globalData.outerLabelColor
        : this.globalData.innerLabelColor;
    } else {
      $statePath.style.fill = this.globalData.fill;
      $stateText.style.fill = isOuterLabel
        ? this.globalData.outerLabelColor
        : this.globalData.innerLabelColor;
    }
    var allShadows = document.querySelectorAll(
      "#" + this.$map.id + " ." + classPrefix + "state-shadow",
    );

    Array.prototype.map.call(
      Array.prototype.slice.call(allShadows),
      function (ele) {
        ele.parentNode.removeChild(ele);
      },
    );
  }

  this.create = function (wrapperId) {
    return new this.mapObject(wrapperId);
  };
  this.mapObject = function (wrapperId) {
    this.$map = global.getEleById(wrapperId);
    this.globalData = createBaseGlobalData();
    this.statesData = createBaseStatesData();

    for (var stateId in this.statesData) {
      if (!this.statesData.hasOwnProperty(stateId)) {
        continue;
      }
      (function (stateId) {
        var $stateLink = global.getEleByQuery(
          "#" + this.$map.id + " ." + global.stateIdToDomClass(stateId),
        );
        var self = this;
        $stateLink.addEventListener("mouseover", function (e) {
          stateOn.call(self, stateId);
        });
        $stateLink.addEventListener("mouseout", function (e) {
          stateOff.call(self, stateId);
        });
        $stateLink = null;
      }).call(this, stateId);
    }
    global
      .getEleByQuery("#" + this.$map.id + " ." + classPrefix + "blur-filter")
      .setAttribute("id", this.$map.id + "-blur-filter");
  };
  this.mapObject.prototype.getDomId = function () {
    return this.$map.id;
  };
  this.mapObject.prototype.draw = function () {
    this.$map.style.width = this.globalData.width + this.globalData.widthUnits;
    this.$map.style.backgroundColor = this.globalData.backgroundFill;
    this.$map.style.fontFamily = this.globalData.fontName;
    this.$map.style.fontSize = this.globalData.fontSize;
    global.getEleByQuery(
      "#" + this.$map.id + " ." + classPrefix + "title",
    ).textContent = this.globalData.mapTitle;
    if (
      this.globalData.creditLink != null &&
      this.globalData.creditLink != ""
    ) {
      global.getEleByQuery(
        "#" + this.$map.id + " ." + classPrefix + "credit-link",
      ).innerHTML = '<a target="_blank" href="' + creditLinkUrl + '"></a>';
      global.getEleByQuery(
        "#" + this.$map.id + " ." + classPrefix + "credit-link a",
      ).textContent = this.globalData.creditLink;
    } else {
      global.getEleByQuery(
        "#" + this.$map.id + " ." + classPrefix + "credit-link",
      ).innerHTML = "";
    }
    for (var stateId in this.statesData) {
      if (!this.statesData.hasOwnProperty(stateId)) {
        continue;
      }
      var stateDomClass = global.stateIdToDomClass(stateId);
      var $stateTitle = global.getEleByQuery(
        "#" + this.$map.id + " ." + stateDomClass + " title",
      );
      var $stateDescription = global.getEleByQuery(
        "#" + this.$map.id + " ." + stateDomClass + " desc",
      );
      $stateTitle.textContent = this.statesData[stateId].title;
      $stateDescription.textContent = this.statesData[stateId].description;
      var $statePath = global.getEleByQuery(
        "#" + this.$map.id + " ." + stateDomClass + " path",
      );
      $statePath.style.stroke = this.globalData.borderStroke;
      if (this.globalData.borderType != null) {
        $statePath.style.strokeDasharray = this.globalData.borderType;
      } else {
        $statePath.style.strokeDasharray = "none";
      }
      if (this.statesData[stateId].isDisabled) {
        $statePath.style.fill = this.globalData.disabledFill;
      } else if (
        this.statesData[stateId].overrideFillEnabled &&
        this.statesData[stateId].overrideFill != null
      ) {
        $statePath.style.fill = this.statesData[stateId].overrideFill;
      } else {
        $statePath.style.fill = this.globalData.fill;
      }
      var $allLabels = document.querySelectorAll(
        "#" + this.$map.id + " ." + stateDomClass + " text",
      );
      for (var i = 0; i < $allLabels.length; ++i) {
        $allLabels.item(i).style.fill = this.globalData.innerLabelColor;
      }

      this.wireStateLink(stateId, false);
    }

    var $outerLabels = document.querySelectorAll(
      "#" + this.$map.id + " ." + classPrefix + "outer-label",
    );

    for (var i = 0; i < $outerLabels.length; ++i) {
      $outerLabels.item(i).style.fill = this.globalData.outerLabelColor;
    }
    if (this.globalData.showLinksList) {
      this.displayMapLinksList();
    } else {
      global.getEleByQuery(
        "#" + this.$map.id + " ." + classPrefix + "listview",
      ).innerHTML = "";
    }
    this.$map.style.display = "block";
  };
  this.mapObject.prototype.getGlobalData = function () {
    return this.globalData;
  };

  this.mapObject.prototype.getStatesData = function () {
    return this.statesData;
  };

  this.mapObject.prototype.setGlobalData = function (data) {
    for (var setting in this.globalData) {
      if (
        !this.globalData.hasOwnProperty(setting) ||
        !data.hasOwnProperty(setting)
      ) {
        continue;
      }
      this.globalData[setting] = data[setting];
    }
  };

  this.mapObject.prototype.setStatesData = function (data) {
    for (var state in this.statesData) {
      if (
        !this.statesData.hasOwnProperty(state) ||
        !data.hasOwnProperty(state)
      ) {
        continue;
      }
      for (var setting in this.statesData[state]) {
        if (
          !this.statesData[state].hasOwnProperty(setting) ||
          !data[state].hasOwnProperty(setting)
        ) {
          continue;
        }
        this.statesData[state][setting] = data[state][setting];
      }
    }
  };

  this.mapObject.prototype.wireStateLink = function (
    stateId,
    addLiveClassName,
    linkType,
  ) {
    var clickFn = null;
    linkType = linkType ? linkType : "";
    var $stateLink = global.getEleByQuery(
      "#" + this.$map.id + " ." + global.stateIdToDomClass(stateId) + linkType,
    );
    if (this.statesData[stateId].cssClass != null) {
      $stateLink.setAttribute(
        "class",
        $stateLink.getAttribute("class") +
          " " +
          this.statesData[stateId].cssClass,
      );
    }
    if (this.statesData[stateId].isDisabled) {
      clickFn = null;
    } else if (this.statesData[stateId].linkUrl != null) {
      var self = this;
      clickFn = function (e) {
        var isPop = false;
        if (self.statesData[stateId].overridePopLink != null) {
          isPop = self.statesData[stateId].overridePopLink;
        } else if (self.globalData.popLink) {
          isPop = true;
        }
        if (isPop) {
          window.open(self.statesData[stateId].linkUrl);
        } else {
          document.location.href = self.statesData[stateId].linkUrl;
        }
      };
    } else if (this.globalData.globalLinkUrl != null) {
      var self = this;
      clickFn = function (e) {
        var normalizedUrl = self.globalData.globalLinkUrl.replaceAll(
          "@state",
          stateId,
        );
        var isPop = false;
        if (self.statesData[stateId].overridePopLink != null) {
          isPop = self.statesData[stateId].overridePopLink;
        } else if (self.globalData.popLink) {
          isPop = true;
        }
        if (isPop) {
          window.open(normalizedUrl);
        } else {
          document.location.href = normalizedUrl;
        }
      };
    } else if (this.globalData.globalJsCallback != null) {
      var self = this;
      clickFn = function (e) {
        var fn = window[self.globalData.globalJsCallback];
        if (typeof fn == "function") {
          fn(stateId);
        } else {
          console.log(
            "Unable to execute function: " +
              self.globalData.globalJsCallback +
              '("' +
              stateId +
              '")',
          );
        }
      };
    }

    $stateLink.onclick = clickFn;

    if (addLiveClassName) {
      var liveLinkClassName = classPrefix + "live-link";
      $stateLink.className = $stateLink.className.replace(
        " " + liveLinkClassName,
        "",
      );
      if (clickFn != null) {
        $stateLink.className = $stateLink.className + " " + liveLinkClassName;
      }
    }
  };
  this.mapObject.prototype.displayMapLinksList = function () {
    var $linkList = global.getEleByQuery(
      "#" + this.$map.id + " ." + classPrefix + "listview",
    );
    var allListsHtml = "";
    var stateIds = [];
    for (var stateId in this.statesData) {
      if (!this.statesData.hasOwnProperty(stateId)) {
        continue;
      }
      stateIds.push(stateId);
    }
    var widthPercent = Math.floor(100 / maxTableColumns);
    var itemsPerList = Math.ceil(stateCount / maxTableColumns);
    var sliceStart = 0;
    for (var i = 0; i < maxTableColumns; ++i) {
      var slicedIds = stateIds.slice(sliceStart, sliceStart + itemsPerList);
      sliceStart += itemsPerList;
      if (slicedIds.length > 0) {
        var ul = document.createElement("UL");
        ul.style.maxWidth = widthPercent + "%";
        for (var x = 0; x < slicedIds.length; ++x) {
          var li = document.createElement("LI");
          li.appendChild(document.createElement("SPAN"));
          var a = document.createElement("A");
          a.className =
            classPrefix + "state-" + slicedIds[x].toLowerCase() + "-listview";
          a.textContent = this.statesData[slicedIds[x]].title;
          li.appendChild(a);
          ul.appendChild(li);
        }
        $linkList.appendChild(ul);
      }
    }

    for (var stateId in this.statesData) {
      if (!this.statesData.hasOwnProperty(stateId)) {
        continue;
      }
      this.wireStateLink(stateId, true, "-listview");
    }
  };

  if (typeof exports !== "undefined") {
    module.exports = this;
  }
}).apply(ClickableMap);
var myUsaMap = ClickableMap.create("cmm-usa");
myUsaMap.setGlobalData({
  version: "1.0.0",
  width: "650",
  widthUnits: "px",
  fontSize: "12px",
  fontName: "Verdana",
  fill: "#ffffff",
  hoverFill: "#ffffff",
  disabledFill: "#c2c2c2",
  backgroundFill: "#ffffff",
  innerLabelColor: "#000000",
  outerLabelColor: "#000000",
  hoverLabelColor: "#000000",
  borderType: null,
  borderStroke: "#000000",
  enableShadows: true,
  popLink: false,
  showStateTitleAndDescOnHover: true,
  showLinksList: false,
  globalLinkUrl: null,
  globalJsCallback: null,
  mapTitle: "",
  creditLink: "",
});
myUsaMap.setStatesData({
  AL: {
    fullName: "Alabama",
    title: "Alabama",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  AK: {
    fullName: "Alaska",
    title: "Alaska",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  AZ: {
    fullName: "Arizona",
    title: "Arizona",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  AR: {
    fullName: "Arkansas",
    title: "Arkansas",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  CA: {
    fullName: "California",
    title: "California",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  CO: {
    fullName: "Colorado",
    title: "Colorado",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  CT: {
    fullName: "Connecticut",
    title: "Connecticut",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  DE: {
    fullName: "Delaware",
    title: "Delaware",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  DC: {
    fullName: "District Of Columbia",
    title: "District Of Columbia",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  FL: {
    fullName: "Florida",
    title: "Florida",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  GA: {
    fullName: "Georgia",
    title: "Georgia",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  HI: {
    fullName: "Hawaii",
    title: "Hawaii",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  ID: {
    fullName: "Idaho",
    title: "Idaho",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  IL: {
    fullName: "Illinois",
    title: "Illinois",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  IN: {
    fullName: "Indiana",
    title: "Indiana",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  IA: {
    fullName: "Iowa",
    title: "Iowa",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  KS: {
    fullName: "Kansas",
    title: "Kansas",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  KY: {
    fullName: "Kentucky",
    title: "Kentucky",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  LA: {
    fullName: "Louisiana",
    title: "Louisiana",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  ME: {
    fullName: "Maine",
    title: "Maine",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  MD: {
    fullName: "Maryland",
    title: "Maryland",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  MA: {
    fullName: "Massachusetts",
    title: "Massachusetts",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  MI: {
    fullName: "Michigan",
    title: "Michigan",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  MN: {
    fullName: "Minnesota",
    title: "Minnesota",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  MS: {
    fullName: "Mississippi",
    title: "Mississippi",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  MO: {
    fullName: "Missouri",
    title: "Missouri",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  MT: {
    fullName: "Montana",
    title: "Montana",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  NE: {
    fullName: "Nebraska",
    title: "Nebraska",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  NV: {
    fullName: "Nevada",
    title: "Nevada",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  NH: {
    fullName: "New Hampshire",
    title: "New Hampshire",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  NJ: {
    fullName: "New Jersey",
    title: "New Jersey",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  NM: {
    fullName: "New Mexico",
    title: "New Mexico",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  NY: {
    fullName: "New York",
    title: "New York",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  NC: {
    fullName: "North Carolina",
    title: "North Carolina",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  ND: {
    fullName: "North Dakota",
    title: "North Dakota",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  OH: {
    fullName: "Ohio",
    title: "Ohio",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  OK: {
    fullName: "Oklahoma",
    title: "Oklahoma",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  OR: {
    fullName: "Oregon",
    title: "Oregon",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  PA: {
    fullName: "Pennsylvania",
    title: "Pennsylvania",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  RI: {
    fullName: "Rhode Island",
    title: "Rhode Island",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  SC: {
    fullName: "South Carolina",
    title: "South Carolina",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  SD: {
    fullName: "South Dakota",
    title: "South Dakota",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  TN: {
    fullName: "Tennessee",
    title: "Tennessee",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  TX: {
    fullName: "Texas",
    title: "Texas",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  UT: {
    fullName: "Utah",
    title: "Utah",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  VT: {
    fullName: "Vermont",
    title: "Vermont",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  VA: {
    fullName: "Virginia",
    title: "Virginia",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  WA: {
    fullName: "Washington",
    title: "Washington",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  WV: {
    fullName: "West Virginia",
    title: "West Virginia",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  WI: {
    fullName: "Wisconsin",
    title: "Wisconsin",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
  WY: {
    fullName: "Wyoming",
    title: "Wyoming",
    description: null,
    longDescription: null,
    linkUrl: null,
    isDisabled: false,
    isHovering: false,
    cssClass: null,
    overrideFill: null,
    overrideFillEnabled: false,
    overrideHoverFill: null,
    overrideHoverFillEnabled: false,
    overridePopLink: null,
  },
});
myUsaMap.draw();
