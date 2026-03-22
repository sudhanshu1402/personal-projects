# Photo Manager

## Overview
**Photo Manager** is a **Hard** difficulty project implemented in **Java**.

## 📂 Project Structure
The following directory structure visualizes the file organization of this project.

```text
Photo Manager
├── app
│   ├── build.gradle
│   ├── proguard-rules.pro
│   └── src
│       ├── androidTest
│       │   └── java
│       │       └── com
│       │           └── example
│       │               └── sudhanshusingh
│       │                   └── manageit
│       │                       └── ExampleInstrumentedTest.java
│       ├── main
│       │   ├── AndroidManifest.xml
│       │   ├── ic_check_image-web.png
│       │   ├── ic_launcher-web.png
│       │   ├── java
│       │   │   └── com
│       │   │       └── example
│       │   │           └── sudhanshusingh
│       │   │               └── manageit
│       │   │                   ├── AllImagesFragment.java
│       │   │                   ├── MainActivity.java
│       │   │                   ├── MyAllImagesViewAdapter.java
│       │   │                   ├── MyDataProvider.java
│       │   │                   ├── MyDatabaseHelper.java
│       │   │                   ├── MyTaggedImagesViewAdapter.java
│       │   │                   ├── MyViewAdapter.java
│       │   │                   ├── MyViewPagerAdapter.java
│       │   │                   ├── ScrollingFABBehavior.java
│       │   │                   ├── SelectableAdapter.java
│       │   │                   ├── ShowImagesActivity.java
│       │   │                   └── TaggedImagesFragment.java
│       │   └── res
│       │       ├── drawable
│       │       │   ├── android
│       │       │   │   ├── drawable-hdpi
│       │       │   │   │   └── ic_add_white_24dp.png
│       │       │   │   ├── drawable-mdpi
│       │       │   │   │   └── ic_add_white_24dp.png
│       │       │   │   ├── drawable-xhdpi
│       │       │   │   │   └── ic_add_white_24dp.png
│       │       │   │   ├── drawable-xxhdpi
│       │       │   │   │   └── ic_add_white_24dp.png
│       │       │   │   └── drawable-xxxhdpi
│       │       │   │       └── ic_add_white_24dp.png
│       │       │   ├── drawable-hdpi
│       │       │   │   └── ic_add_white_24dp.png
│       │       │   ├── drawable-mdpi
│       │       │   │   └── ic_add_white_24dp.png
│       │       │   ├── drawable-xhdpi
│       │       │   │   ├── ic_add_white_24dp.png
│       │       │   │   └── ic_search_white_24dp.png
│       │       │   ├── drawable-xxhdpi
│       │       │   │   └── ic_add_white_24dp.png
│       │       │   ├── drawable-xxxhdpi
│       │       │   │   └── ic_add_white_24dp.png
│       │       │   ├── ic_add_circle_outline_black_24dp.png
│       │       │   ├── ic_add_white_24dp.png
│       │       │   ├── ic_close_24dp.xml
│       │       │   └── ic_search_white_24dp.png
│       │       ├── drawable-hdpi
│       │       │   ├── ic_action_name.png
│       │       │   ├── ic_back.png
│       │       │   ├── ic_camera.png
│       │       │   ├── ic_check_mark.png
│       │       │   ├── ic_image_black.png
│       │       │   └── ic_tagcount.png
│       │       ├── drawable-mdpi
│       │       │   ├── ic_action_name.png
│       │       │   ├── ic_back.png
│       │       │   ├── ic_camera.png
│       │       │   ├── ic_check_mark.png
│       │       │   ├── ic_image_black.png
│       │       │   └── ic_tagcount.png
│       │       ├── drawable-xhdpi
│       │       │   ├── ic_action_name.png
│       │       │   ├── ic_back.png
│       │       │   ├── ic_camera.png
│       │       │   ├── ic_check_mark.png
│       │       │   ├── ic_delete_white_24dp.png
│       │       │   ├── ic_image_black.png
│       │       │   ├── ic_share_white_24dp.png
│       │       │   └── ic_tagcount.png
│       │       ├── drawable-xxhdpi
│       │       │   ├── ic_action_name.png
│       │       │   ├── ic_back.png
│       │       │   ├── ic_camera.png
│       │       │   ├── ic_check_mark.png
│       │       │   ├── ic_image_black.png
│       │       │   ├── ic_light_icon_background.png
│       │       │   ├── ic_tagcount.png
│       │       │   └── white_tag.png
│       │       ├── layout
│       │       │   ├── activity_main.xml
│       │       │   ├── activity_show_images.xml
│       │       │   ├── all_images_single_grid.xml
│       │       │   ├── content_main.xml
│       │       │   ├── content_show_images.xml
│       │       │   ├── fragment_all_images.xml
│       │       │   ├── fragment_tagged_images.xml
│       │       │   ├── single_image_layout_row.xml
│       │       │   └── single_row_layout.xml
│       │       ├── menu
│       │       │   ├── contextual_menu.xml
│       │       │   ├── images_menu.xml
│       │       │   └── menu_main.xml
│       │       ├── mipmap-hdpi
│       │       │   ├── ic_check_image.png
│       │       │   └── ic_launcher.png
│       │       ├── mipmap-mdpi
│       │       │   ├── ic_check_image.png
│       │       │   └── ic_launcher.png
│       │       ├── mipmap-xhdpi
│       │       │   ├── ic_check_image.png
│       │       │   └── ic_launcher.png
│       │       ├── mipmap-xxhdpi
│       │       │   ├── ic_check_image.png
│       │       │   └── ic_launcher.png
│       │       ├── mipmap-xxxhdpi
│       │       │   ├── ic_check_image.png
│       │       │   └── ic_launcher.png
│       │       ├── provider_paths.xml
│       │       ├── values
│       │       │   ├── colors.xml
│       │       │   ├── dimens.xml
│       │       │   ├── strings.xml
│       │       │   └── styles.xml
│       │       ├── values-v21
│       │       │   └── styles.xml
│       │       ├── values-w820dp
│       │       │   └── dimens.xml
│       │       └── xml
│       │           └── provider_paths.xml
│       └── test
│           └── java
│               └── com
│                   └── example
│                       └── sudhanshusingh
│                           └── manageit
│                               └── ExampleUnitTest.java
├── build.gradle
├── gradle
│   └── wrapper
│       ├── gradle-wrapper.jar
│       └── gradle-wrapper.properties
├── gradle.properties
├── gradlew
├── gradlew.bat
└── settings.gradle

```

## 📐 Components
Visual representation of the primary files in this project:

```mermaid
graph TD
    Photo Manager[Photo Manager]
    Photo_Manager --> gradlew(gradlew)
    Photo_Manager --> build_gradle(build.gradle)
    Photo_Manager --> gradle_properties(gradle.properties)
    Photo_Manager --> gradlew_bat(gradlew.bat)
    Photo_Manager --> settings_gradle(settings.gradle)
```

## Features
- Implements core logic for Photo Manager.
- Structured for scalability and readability.
- Demonstrates **Java** best practices for **Hard** complexity.

## How to Run
1. Navigate to the project directory:
   ```bash
   cd Photo Manager
   ```
2. Check the source code for entry points (e.g., `main` run command).
