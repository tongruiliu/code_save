import os
import warnings
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning
from playwright.sync_api import sync_playwright

# Suppress XMLParsedAsHTMLWarning to avoid noise when parsing SVGs with HTML parser
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

class Blackboard:
    """
    一个用于动态管理和渲染HTML/SVG内容的“黑板”。
    它通过一系列操作（如插入、修改、删除）来维护一个HTML字符串状态，
    并能将该状态渲染成图片。
    """
    def __init__(self, initial_svg=None):
        """
        初始化黑板，设置一个基本的HTML5文档结构。
        所有动态内容将被插入到<body>标签内。
        """
        try:
            BeautifulSoup("<b></b>", "lxml")
            self.parser = "lxml"
        except Exception:
            self.parser = "html.parser"

        self.state = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NoteBook State</title>
    <style>
    /* 1. 全局基础设置，保持不变 */
    *,
    *::before,
    *::after {
      box-sizing: border-box;
    }

    body {
      font-family: sans-serif;
      line-height: 1.6;
      background-color: #f4f7f9;
      color: #333;
      margin: 0;
      padding: 20px;
    }

    /* 使用元素选择器代替 .main-container */
    main {
      max-width: 800px;
      margin: 20px auto;
      padding: 20px;
      background-color: #ffffff;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    h1, h2 {
      border-bottom: 2px solid #e0e0e0;
      padding-bottom: 10px;
      color: #1a2c3b;
    }
    
    p {
      margin-top: 0;
    }
    
    /* 
      使用 section 元素包裹每个演示区域，
      这样我们可以用 :nth-of-type() 来区分它们。
    */
    section {
      margin-bottom: 30px;
    }

    /* --- 示例样式 (无Class版) --- */

    /* 2. 宽度控制示例 */
    /* 选择第一个 section 里的直接子 div (作为父容器) */
    section:nth-of-type(1) > div {
      padding: 20px;
      border: 2px dashed #666;
      background-color: #f0f0f0;
    }

    /* 选择父容器内的所有直接子 div (作为子盒子) */
    section:nth-of-type(1) > div > div {
      padding: 15px;
      border: 1px solid;
      color: #fff;
      margin-bottom: 10px;
    }

    /* 用伪类选择第一个子盒子 (良好实践) */
    section:nth-of-type(1) > div > div:first-of-type {
      background-color: #009E5F;
      border-color: #007A4B;
      /* 默认 width: auto，无需设置 */
    }
    
    /* 用伪类选择最后一个子盒子 (不良实践) */
    section:nth-of-type(1) > div > div:last-of-type {
      background-color: #ED2633;
      border-color: #B81E27;
      width: 550px; /* 故意设置固定宽度以演示溢出 */
    }


    /* 3. 响应式图片示例 */
    /* 给所有在 main 元素内的图片应用响应式规则 */
    main img {
      max-width: 100%;
      height: auto;
      display: block;
    }
    /* 为图片的容器(figure)添加边框 */
    main figure {
      border: 2px solid #1377EB;
      padding: 10px;
      margin: 0 0 20px 0; /* 重置 figure 的默认 margin */
    }


    /* 4. 高度自适应与最小高度示例 */
    /* 选择第三个 section 里的 div (作为 flex 容器) */
    section:nth-of-type(3) > div {
      display: flex;
      gap: 20px;
    }
    
    /* 选择 flex 容器内的所有 article 元素 (作为卡片) */
    section:nth-of-type(3) > div > article {
      flex: 1;
      border: 1px solid #ccc;
      padding: 15px;
      border-radius: 5px;
      background: #fafafa;
      min-height: 120px; /* 关键规则 */
    }


    /* 5. 长文本换行处理示例 */
    /* 选择第四个 section 里的 div (作为文本容器) */
    section:nth-of-type(4) > div {
      border: 2px solid #f2994a;
      padding: 15px;
      background-color: #fff8f2;
      overflow-wrap: break-word; /* 关键规则 */
      word-wrap: break-word; /* 兼容性写法 */
    }

    /* 选择文本容器里的最后一个 p 元素来改变其样式 */
    section:nth-of-type(4) > div > p:last-of-type {
      font-size: 0.9em;
      color: #777;
    }

  </style>
</head>
<body>
  <div id="root"></div>
</body>
</html>
"""
        if initial_svg:
            # Ensure initial_svg has an ID if possible, to help the model reference it
            try:
                soup_temp = BeautifulSoup(initial_svg, self.parser)
                tag = soup_temp.find()
                if tag and not tag.get('id'):
                    # We can't easily modify the string without parsing, but update_state parses it.
                    # So we'll just pass it, and let update_state handle insertion.
                    # But to be safe, let's try to inject it if we can, or just rely on the model to find it.
                    # Actually, let's just insert it. The model can use 'modify_element' if it finds the ID,
                    # or we can force an ID.
                    pass
            except:
                pass
            self.update_state(action="insert_element", attrs={"fragment": initial_svg, "rootId": None})

    def _allow_external_images(self) -> bool:
        return os.environ.get("BLACKBOARD_ALLOW_EXTERNAL_IMAGES", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    def _is_external_url(self, url: str) -> bool:
        u = (url or "").strip().lower()
        return u.startswith("http://") or u.startswith("https://")

    def _sanitize_external_images(self, fragment_soup: BeautifulSoup) -> None:
        """Remove/replace external image URLs to keep rendering offline-safe.

        The notebook is meant to render deterministically from provided inputs.
        External URLs (e.g., imgur) often fail in sandboxed environments and
        produce placeholder images like "The image you are requesting does not exist".

        Set env var BLACKBOARD_ALLOW_EXTERNAL_IMAGES=1 to disable this sanitization.
        """
        if self._allow_external_images():
            return

        # HTML <img src="...">
        for img in list(fragment_soup.find_all("img")):
            src = img.get("src")
            if isinstance(src, str) and self._is_external_url(src):
                replacement = fragment_soup.new_tag(
                    "div",
                    attrs={
                        "style": "color:#ED2633;font-size:14px;"
                        "padding:8px 0;"
                        "word-break:break-word;"
                    },
                )
                replacement.string = f"[blocked external image] {src}"
                img.replace_with(replacement)

        # SVG <image href="..."> / xlink:href
        for svg_img in list(fragment_soup.find_all("image")):
            href = svg_img.get("href") or svg_img.get("xlink:href")
            if isinstance(href, str) and self._is_external_url(href):
                parent = svg_img.parent
                if parent and hasattr(parent, "new_tag"):
                    try:
                        x = svg_img.get("x") or "0"
                        y = svg_img.get("y") or "0"
                        text = fragment_soup.new_tag("text")
                        text["x"] = str(x)
                        text["y"] = str(y)
                        text["fill"] = "#ED2633"
                        text.string = "[blocked external svg image]"
                        svg_img.replace_with(text)
                    except Exception:
                        svg_img.decompose()
                else:
                    svg_img.decompose()

    def _extract_fragment_root(self, fragment_str: str):
        """Extract the intended root tag from a fragment.

        Using HTML parser for SVG fragments often introduces wrapper nodes like
        <html><body>...</body></html>. This helper unwraps those wrappers and
        returns the first meaningful element.
        """
        fragment_soup = BeautifulSoup(fragment_str, self.parser)
        self._sanitize_external_images(fragment_soup)

        # Prefer body children if HTML wrappers exist.
        if fragment_soup.body:
            for child in fragment_soup.body.contents:
                if isinstance(child, Tag):
                    return child

        # Fallback: first tag that is not a wrapper.
        for tag in fragment_soup.find_all(True):
            if tag.name not in {"html", "body"}:
                return tag
        return None

    def update_state(self, action: str, attrs: dict):
        """
        根据指定的操作和属性来更新黑板的HTML状态。

        Args:
            action (str): 操作类型，如 "insert", "modify", "remove", "replace", "clear"。
            attrs (dict): 操作所需的参数字典。
        """
        # 使用self.parser解析器以获得更好的XML/SVG兼容性
        soup = BeautifulSoup(self.state, self.parser)
        
        # --- 1. insert: 插入新元素 ---
        # 对应工具: insert_element
        if action == "insert_element":
            fragment_str = attrs.get('fragment')
            if not fragment_str:
                print("警告 (insert): 'fragment' 不能为空。")
                return

            # Parse fragment and sanitize external resources (offline-safe rendering)
            # while avoiding accidental html/body wrappers for svg/text snippets.
            new_element = self._extract_fragment_root(fragment_str)
            if not new_element:
                print(f"警告 (insert): 无法解析提供的fragment: {fragment_str}")
                return
            
            # Auto-assign ID if missing (crucial for initial_svg)
            if not new_element.get('id'):
                new_element['id'] = "initial_svg" if new_element.name == "svg" else f"elem_{len(soup.find_all())}"

            # 确定父节点，默认为<body>
            root_id = attrs.get('rootId')
            parent_element = soup.find(id=root_id) if root_id else soup.body
            if not parent_element:
                if root_id == 'root':
                    print(f"警告 (insert): 找不到ID为 'root' 的父节点，正在自动创建...")
                    new_root = soup.new_tag("div", id="root")
                    soup.body.append(new_root)
                    parent_element = new_root
                else:
                    print(f"警告 (insert): 找不到ID为 '{root_id}' 的父节点，将插入到<body>中。")
                    parent_element = soup.body

            # 确定插入位置
            before_id = attrs.get('beforeId')
            if before_id:
                before_element = soup.find(id=before_id)
                if before_element:
                    before_element.insert_before(new_element)
                else:
                    print(f"警告 (insert): 找不到ID为 '{before_id}' 的兄弟节点，将追加到末尾。")
                    parent_element.append(new_element)
            else:
                parent_element.append(new_element)

        # --- 2. modify: 修改已有元素 ---
        # 对应工具: modify_element
        elif action == "modify_element":
            target_id = attrs.get('targetId')
            attributes_to_update = attrs.get('attrs')
            if not target_id or not attributes_to_update:
                print("警告 (modify): 'targetId' 和 'attrs' 不能为空。")
                return

            target_element = soup.find(id=target_id)
            if target_element:
                for key, value in attributes_to_update.items():
                    # 特殊处理 'text'，用于修改标签内的文本内容
                    if key == 'text':
                        target_element.string = str(value)
                    else:
                        target_element[key] = str(value)
            else:
                print(f"警告 (modify): 找不到ID为 '{target_id}' 的元素。")

        # --- 3. remove: 删除已有元素 ---
        # 对应工具: remove_element
        elif action == "remove_element":
            target_id = attrs.get('targetId')
            if not target_id:
                print("警告 (remove): 'targetId' 不能为空。")
                return
                
            target_element = soup.find(id=target_id)
            if target_element:
                target_element.decompose()  # decompose()会彻底移除标签
            else:
                print(f"警告 (remove): 找不到ID为 '{target_id}' 的元素。")

        # --- 4. clear: 清空所有元素 ---
        # 对应工具: clear_blackboard
        elif action == "clear_element" or action == "clear":
            if soup.body:
                soup.body.clear() # clear()会移除body标签的所有子节点
                # 重新添加 root div
                new_root = soup.new_tag("div", id="root")
                soup.body.append(new_root)
            else:
                # 如果没有body，则重新创建基础结构
                self.__init__()
                return

        # --- 额外操作: 替换元素 (基于工具 `replace_element`) ---
        elif action == "replace_element":
            target_id = attrs.get('targetId')
            fragment_str = attrs.get('fragment')
            if not target_id or not fragment_str:
                print("警告 (replace): 'targetId' 和 'fragment' 不能为空。")
                return
            
            target_element = soup.find(id=target_id)
            if target_element:
                new_element = self._extract_fragment_root(fragment_str)
                if new_element:
                    target_element.replace_with(new_element)
                else:
                    print(f"警告 (replace): 无法解析提供的fragment: {fragment_str}")
            else:
                print(f"警告 (replace): 找不到ID为 '{target_id}' 的元素。")

        else:
            print(f"错误: 未知的操作 '{action}'")
            return
            
        # 将修改后的BeautifulSoup对象转换回字符串，并更新状态
        self.state = str(soup)

    def render_state(self, output_path="output.png"):
        """
        --- 5. render_state: 渲染当前状态 ---
        将当前的HTML状态渲染成一张高清图片。
        
        *修改*: 为了防止画布过长导致模型看不清，这里只截取body中最后3个元素进行渲染。
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        try:
            # 使用临时soup处理，不影响真实state
            soup = BeautifulSoup(self.state, self.parser)
            
            # 只保留body中最后3个标签元素
            if soup.body:
                # 筛选出Tag类型的子节点（忽略换行符等文本节点）
                children = [child for child in soup.body.contents if child.name]
                if len(children) > 3:
                    for child in children[:-3]:
                        child.decompose()

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(
                    viewport={"width": 500, "height": 500}, 
                    device_scale_factor=2 # 使用2倍缩放以获得更清晰的图像
                )
                # 使用处理后的HTML内容
                page.set_content(str(soup))
                page.screenshot(path=output_path, full_page=True)
                browser.close()
                print(f"高清截图已保存: {output_path}")
                return "tool execute success"
        except Exception as e:
            print(f"tool execute failed: {e}")
            return f"tool execute failed: {e}"
            




if __name__ == '__main__':
    
    blackboard = Blackboard()

    svg_canvas_fragment = '<svg id=\"objs1\" width=\"500\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">\n<rect id=\"large_red\" x=\"20\" y=\"50\" width=\"50\" height=\"50\" fill=\"#ED2633\" stroke=\"#000\" stroke-width=\"1\"/>\n<rect id=\"small_yellow\" x=\"120\" y=\"70\" width=\"30\" height=\"30\" fill=\"#FFD700\" stroke=\"#000\" stroke-width=\"1\"/>\n<rect id=\"small_red\" x=\"200\" y=\"70\" width=\"30\" height=\"30\" fill=\"#ED2633\" stroke=\"#000\" stroke-width=\"1\"/>\n<text id=\"lbl1\" x=\"20\" y=\"40\" fill=\"#000\">Large red rubber</text>\n<text id=\"lbl2\" x=\"120\" y=\"60\" fill=\"#000\">Small yellow matte</text>\n<text id=\"lbl3\" x=\"200\" y=\"60\" fill=\"#000\">Small red matte</text>\n</svg>'
    blackboard.update_state(
        action="insert_element",
        attrs={'fragment': svg_canvas_fragment}
    )
    blackboard.render_state("blackboard_output/1_text_added.png")
    print("插入画布后:", blackboard.state)

    svg_canvas_fragment = {'stroke': '#ED2633', 'stroke-width': '3'}
    blackboard.update_state(
        action="modify_element",
        attrs={
            'targetId': 'large_red',
            'attrs': svg_canvas_fragment
        }
    )
    blackboard.render_state("blackboard_output/1_text_added2.png")
    print("插入画布后:", blackboard.state)




# if __name__ == '__main__':
#     # 1. 创建一个黑板实例
#     print("--- 1. 初始化黑板 ---")
#     blackboard = Blackboard()
    
#     # 渲染初始的空状态
#     blackboard.render_state("blackboard_output/0_initial_state.png")
#     print("初始状态:", blackboard.state)


#     # 2. 插入一个SVG画布作为绘图区域
#     print("\n--- 2. 插入一个SVG画布 ---")
    # svg_canvas_fragment = "<svg id='drawing-area' width='500' height='400' style='border: 2px solid #ccc;'></svg>"
    # blackboard.update_state(
    #     action="insert",
    #     attrs={'fragment': svg_canvas_fragment}
    # )
    # blackboard.render_state("blackboard_output/1_svg_canvas_added.png")
    # print("插入画布后:", blackboard.state)


#     # 3. 在SVG中插入一个蓝色圆形
#     print("\n--- 3. 在SVG中插入一个蓝色圆形 ---")
#     circle_fragment = "<circle id='my-circle' cx='100' cy='100' r='80' fill='blue' />"
#     blackboard.update_state(
#         action="insert",
#         attrs={
#             'fragment': circle_fragment,
#             'rootId': 'drawing-area'  # 指定父节点
#         }
#     )
#     blackboard.render_state("blackboard_output/2_circle_added.png")
#     print("插入圆形后:", blackboard.state)


#     # 4. 修改圆形的属性，把它变大变红
#     print("\n--- 4. 修改圆形属性 (颜色和半径) ---")
#     blackboard.update_state(
#         action="modify",
#         attrs={
#             'targetId': 'my-circle',
#             'attrs': {'fill': 'red', 'r': '80'}
#         }
#     )
#     blackboard.render_state("blackboard_output/3_circle_modified.png")
#     print("修改圆形后:", blackboard.state)


#     # 5. 插入一个矩形，并放在圆形的前面
#     print("\n--- 5. 插入一个绿色矩形 (在圆形之前) ---")
#     rect_fragment = "<rect id='my-rect' x='200' y='150' width='150' height='100' fill='green' />"
#     blackboard.update_state(
#         action="insert",
#         attrs={
#             'fragment': rect_fragment,
#             'rootId': 'drawing-area',
#             'beforeId': 'my-circle' # 指定插入到哪个元素之前
#         }
#     )
#     blackboard.render_state("blackboard_output/4_rect_added.png")
#     print("插入矩形后:", blackboard.state)


#     # 6. 使用 replace 操作将圆形替换成一段文字
#     print("\n--- 6. 将红色圆形替换为文本 ---")
#     text_fragment = "<text id='greeting' x='80' y='120' font-family='Verdana' font-size='35' fill='orange'>Hello!</text>"
#     blackboard.update_state(
#         action="replace",
#         attrs={
#             'targetId': 'my-circle',
#             'fragment': text_fragment
#         }
#     )
#     blackboard.render_state("blackboard_output/5_circle_replaced.png")
#     print("替换圆形后:", blackboard.state)


#     # 7. 删除矩形
#     print("\n--- 7. 删除绿色矩形 ---")
#     blackboard.update_state(
#         action="remove",
#         attrs={'targetId': 'my-rect'}
#     )
#     blackboard.render_state("blackboard_output/6_rect_removed.png")
#     print("删除矩形后:", blackboard.state)


#     # 8. 清空整个黑板 (body内的所有内容)
#     print("\n--- 8. 清空黑板 ---")
#     blackboard.update_state(action="clear", attrs={})
#     blackboard.render_state("blackboard_output/7_blackboard_cleared.png")
#     print("清空黑板后:", blackboard.state)
    
#     print("\n所有操作完成！请查看 'output' 文件夹中的图片。")
