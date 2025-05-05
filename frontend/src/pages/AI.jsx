import React, { useState, useEffect } from "react"
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import ai_logo from '../assets/ai.svg'
import users_logo from '../assets/consumers.svg'
import products_logo from '../assets/products.svg'
import recommendation_logo from '../assets/recommendation.svg'
import UserList from './aisubpages/UserList'
import ProductList from './aisubpages/ProductList'
import Recommendation from './aisubpages/Recommendation'
import Evaluation from './aisubpages/Evaluation'

const SubTab = ({logo, text, clicked}) => {
    return (
        <div onClick={clicked} className="flex flex-row justify-center w-auto h-auto p-2 gap-1 cursor-pointer">
            <img src={logo} alt="AI Logo" className="h-8 aspect-w-1 aspect-h-1"/>
            <p className="flex justify-center items-center text-sm text-[var(--text-secondary)] h-full">{text}</p>
        </div>
    )
}

const AI = () => {

    const subPages = [<ProductList/>, <UserList/>, <Recommendation/>, /* <Evaluation/> */]
    const [subpageidx, setSubPage] = useState(0)

    return (
        <main className="flex flex-col h-0 flex-grow bg-[var(--background)]">
            <div className="flex flex-row justify-center w-auto h-auto p-2">
                <SubTab logo={products_logo} text="Product List" clicked={() => setSubPage(0)}/>
                <SubTab logo={users_logo} text="UserList" clicked={() => setSubPage(1)}/>
                <SubTab logo={recommendation_logo} text="Recommendation" clicked={() => setSubPage(2)}/>
                {/* <SubTab logo={ai_logo} text="Evaluation" clicked={() => setSubPage(3)}/> */}
            </div>
            <section className="flex-grow pl-16 pr-16 overflow-y-auto">
                {subPages[subpageidx]}
            </section>
        </main>
    )
}

export default AI