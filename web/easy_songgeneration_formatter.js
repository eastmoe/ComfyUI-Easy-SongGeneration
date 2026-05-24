import { app } from "../../scripts/app.js";

const NODE_NAME = "EasySongGenerationLyricsStyleFormatter";

const lyricGroups = [
    {
        title: "段落结构",
        target: "lyrics",
        mode: "section",
        items: [
            ["短前奏", "intro-short"],
            ["中前奏", "intro-medium"],
            ["主歌", "verse"],
            ["副歌", "chorus"],
            ["桥段", "bridge"],
            ["短间奏", "inst-short"],
            ["中间奏", "inst-medium"],
            ["短尾奏", "outro-short"],
            ["中尾奏", "outro-medium"],
        ],
    },
];

const styleGroups = [
    {
        title: "人声",
        target: "style",
        mode: "style",
        items: [
            ["女声", "female"],
            ["男声", "male"],
            ["混合人声", "mixed"],
            ["女声演唱", "female vocals"],
            ["男声演唱", "male vocals"],
            ["和声", "backing vocals"],
        ],
    },
    {
        title: "曲风",
        target: "style",
        mode: "style",
        items: [
            ["流行", "pop"],
            ["流行抒情", "pop ballad"],
            ["合成器流行", "synth-pop"],
            ["摇滚", "rock"],
            ["R&B", "r&b"],
            ["嘻哈", "hip-hop"],
            ["爵士", "jazz"],
            ["电子", "electronic"],
            ["舞曲流行", "dance-pop"],
            ["民谣", "folk"],
            ["乡村", "country"],
            ["电影感", "cinematic"],
            ["国风", "gufeng"],
            ["中国传统", "chinese traditional"],
        ],
    },
    {
        title: "情绪",
        target: "style",
        mode: "style",
        items: [
            ["充满活力", "energetic"],
            ["振奋", "uplifting"],
            ["浪漫", "romantic"],
            ["忧郁", "melancholic"],
            ["怀旧", "nostalgic"],
            ["心碎", "heartbroken"],
            ["梦幻", "dreamy"],
            ["自信", "confident"],
            ["内省", "introspective"],
            ["愤怒", "angry"],
            ["甜美", "sweet"],
            ["坚韧", "resilient"],
        ],
    },
    {
        title: "乐器",
        target: "style",
        mode: "style",
        items: [
            ["钢琴", "piano"],
            ["木吉他", "acoustic guitar"],
            ["电吉他", "electric guitar"],
            ["贝斯吉他", "bass guitar"],
            ["鼓组", "drum kit"],
            ["鼓机", "drum machine"],
            ["合成器", "synthesizer"],
            ["合成贝斯", "synth bass"],
            ["弦乐组", "string section"],
            ["小提琴", "violin"],
            ["大提琴", "cello"],
            ["萨克斯", "saxophone"],
            ["古筝", "guzheng"],
            ["二胡", "erhu"],
            ["笛子", "dizi"],
        ],
    },
];

const presetGroups = [
    {
        title: "风格预设",
        target: "style",
        mode: "preset",
        items: [
            ["流行抒情", "female, pop ballad, romantic, piano, string section, drum kit"],
            ["摇滚励志", "male, rock, motivational, electric guitar, bass guitar, drum kit"],
            ["甜美合成器流行", "female, synth-pop, sweet, synthesizer, drum machine, bass"],
            ["国风苦甜", "female, gufeng, bittersweet, guzheng, erhu, dizi, percussion"],
            ["自信爵士", "female, jazz, confident, piano, brass section, double bass, drum kit"],
        ],
    },
];

function injectStyle() {
    if (document.getElementById("easy-songgen-formatter-style")) {
        return;
    }
    const style = document.createElement("style");
    style.id = "easy-songgen-formatter-style";
    style.textContent = `
        .easy-songgen-tools {
            box-sizing: border-box;
            width: 100%;
            min-width: 320px;
            padding: 8px 4px 2px;
            color: var(--input-text, #ddd);
            font-family: sans-serif;
        }
        .easy-songgen-tools__group {
            margin: 0 0 8px;
        }
        .easy-songgen-tools__title {
            margin: 0 0 4px;
            color: var(--fg-color, #bbb);
            font-size: 11px;
            font-weight: 600;
            line-height: 1.2;
        }
        .easy-songgen-tools__buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }
        .easy-songgen-tools button {
            border: 1px solid var(--border-color, #555);
            border-radius: 4px;
            background: var(--comfy-input-bg, #2b2b2b);
            color: var(--input-text, #ddd);
            padding: 3px 6px;
            font-size: 11px;
            line-height: 1.25;
            cursor: pointer;
        }
        .easy-songgen-tools button:hover {
            border-color: var(--fg-color, #aaa);
            background: var(--comfy-menu-bg, #333);
        }
    `;
    document.head.appendChild(style);
}

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function findInputElement(widget) {
    return (
        widget?.inputEl ||
        widget?.element?.querySelector?.("textarea,input") ||
        widget?.el?.querySelector?.("textarea,input") ||
        null
    );
}

function setWidgetValue(node, widget, value, cursorStart, cursorEnd) {
    widget.value = value;
    const input = findInputElement(widget);
    if (input) {
        input.value = value;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
        input.focus();
        if (typeof input.setSelectionRange === "function") {
            input.setSelectionRange(cursorStart, cursorEnd);
        }
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function insertAtCursor(node, widgetName, text, mode) {
    const widget = findWidget(node, widgetName);
    if (!widget) {
        return;
    }

    const input = findInputElement(widget);
    const value = String(widget.value ?? "");
    const hasSelection = input && typeof input.selectionStart === "number";
    const start = hasSelection ? input.selectionStart : value.length;
    const end = hasSelection ? input.selectionEnd : value.length;
    const before = value.slice(0, start);
    const after = value.slice(end);
    let insertText = text;

    if (mode === "section") {
        insertText = `[${text}]`;
        if (after && !/^[\s;]/.test(after)) {
            insertText += " ";
        }
        if (before && !/[\s;]$/.test(before)) {
            insertText = " ; " + insertText;
        }
    } else if (mode === "style") {
        insertText = text;
        if (before.trim() && !/[,;\s]$/.test(before)) {
            insertText = ", " + insertText;
        }
        if (after.trim() && !/^\s*[,;.]/.test(after)) {
            insertText += ", ";
        } else if (!after.trim()) {
            insertText += ", ";
        }
    } else if (mode === "preset") {
        insertText = text;
    }

    const nextValue = before + insertText + after;
    const cursor = before.length + insertText.length;
    setWidgetValue(node, widget, nextValue, cursor, cursor);
}

function makeButton(node, item, group) {
    const button = document.createElement("button");
    const label = Array.isArray(item) ? item[0] : item;
    const value = Array.isArray(item) ? item[1] : item;
    button.type = "button";
    button.textContent = label;
    button.title = group.mode === "section" ? `插入 [${value}]` : `插入 ${value}`;
    button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        insertAtCursor(node, group.target, value, group.mode);
    });
    return button;
}

function buildTools(node) {
    injectStyle();
    const root = document.createElement("div");
    root.className = "easy-songgen-tools";

    for (const group of [...lyricGroups, ...styleGroups, ...presetGroups]) {
        const block = document.createElement("div");
        block.className = "easy-songgen-tools__group";

        const title = document.createElement("div");
        title.className = "easy-songgen-tools__title";
        title.textContent = group.title;
        block.appendChild(title);

        const buttons = document.createElement("div");
        buttons.className = "easy-songgen-tools__buttons";
        for (const item of group.items) {
            buttons.appendChild(makeButton(node, item, group));
        }
        block.appendChild(buttons);
        root.appendChild(block);
    }

    return root;
}

function addFallbackButtons(node) {
    for (const group of [...lyricGroups, ...styleGroups, ...presetGroups]) {
        for (const item of group.items) {
            const label = Array.isArray(item) ? item[0] : item;
            const value = Array.isArray(item) ? item[1] : item;
            const name = label;
            node.addWidget("button", name, null, () => insertAtCursor(node, group.target, value, group.mode), {
                serialize: false,
            });
        }
    }
}

function addFormatterTools(node) {
    if (node._easySonggenFormatterTools) {
        return;
    }
    node._easySonggenFormatterTools = true;

    if (typeof node.addDOMWidget === "function") {
        const root = buildTools(node);
        const widget = node.addDOMWidget("songgen_formatter_shortcuts", "div", root, {
            serialize: false,
            hideOnZoom: false,
            getValue: () => "",
            setValue: () => {},
        });
        widget.serialize = false;
        widget.computeSize = (width) => [width, root.offsetHeight + 12];
        node.setSize?.([Math.max(node.size?.[0] ?? 0, 520), node.size?.[1] ?? 0]);
        return;
    }

    addFallbackButtons(node);
}

app.registerExtension({
    name: "easy-songgeneration.lyrics-style-formatter",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            addFormatterTools(this);
        };
    },
});
