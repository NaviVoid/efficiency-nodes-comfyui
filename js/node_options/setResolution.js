// Additional functions and imports
import { app } from "../../../scripts/app.js";
import { findWidgetByName } from "./common/utils.js";

// A mapping for resolutions based on the type of the loader
const RESOLUTIONS = {
    "Efficient Loader": [
        {width: 512, height: 512},
        {width: 512, height: 768},
        {width: 512, height: 640},
        {width: 640, height: 512},
        {width: 640, height: 768},
        {width: 640, height: 640},
        {width: 768, height: 512},
        {width: 768, height: 768},
        {width: 768, height: 640},
    ],
    "Eff. Loader SDXL": [
        {width: 1024, height: 1024},
        {width: 1152, height: 896},
        {width: 896, height: 1152},
        {width: 1216, height: 832},
        {width: 832, height: 1216},
        {width: 1344, height: 768},
        {width: 768, height: 1344},
        {width: 1536, height: 640},
        {width: 640, height: 1536}
    ]
};

// Function to set the resolution of a node
function setNodeResolution(node, width, height) {
    let widthWidget = findWidgetByName(node, "empty_latent_width");
    let heightWidget = findWidgetByName(node, "empty_latent_height");

    if (widthWidget) {
        widthWidget.value = width;
    }

    if (heightWidget) {
        heightWidget.value = height;
    }
}

// Extension Definition
app.registerExtension({
    name: "efficiency.SetResolution",
    getNodeMenuItems(node) {
        const resolutions = RESOLUTIONS[node.comfyClass];
        if (!resolutions) return [];

        const options = resolutions.map(res => ({
            content: `${res.width} x ${res.height}`,
            callback: () => setNodeResolution(node, res.width, res.height)
        }));

        return [{ content: "📐 Set Resolution...", submenu: { options } }];
    },
});
