import React, { useState, useEffect } from "react"
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import ai from '../../assets/ai.svg'
import ProductCard from "../../components/ProductCard"

/*
users: (user_id, user_latents)
products: product_id,product_name,aisle_id,department_id, (product_latents)
orders: order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order
order_products: order_id,product_id,add_to_cart_order,reordered
departments: department_id,department
aisles: aisle_id,aisle
*/
const temps = [
    {
        product_id: 1,
        product_name: 'Hello',
        aisle_id: 1,
        aisle: 'Aisle Lorem Ipsum',
        department_id: 1,
        department: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 1,
        product_name: 'Hello',
        aisle_id: 1,
        aisle: 'Aisle Lorem Ipsum',
        department_id: 1,
        department: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 1,
        product_name: 'Hello',
        aisle_id: 1,
        aisle: 'Aisle Lorem Ipsum',
        department_id: 1,
        department: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 1,
        product_name: 'Hello',
        aisle_id: 1,
        aisle: 'Aisle Lorem Ipsum',
        department_id: 1,
        department: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 1,
        product_name: 'Hello',
        aisle_id: 1,
        aisle: 'Aisle Lorem Ipsum',
        department_id: 1,
        department: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 1,
        product_name: 'Hello',
        aisle_id: 1,
        aisle: 'Aisle Lorem Ipsum',
        department_id: 1,
        department: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 1,
        product_name: 'Hello',
        aisle_id: 1,
        aisle: 'Aisle Lorem Ipsum',
        department_id: 1,
        department: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 1,
        product_name: 'Hello',
        aisle_id: 1,
        aisle: 'Aisle Lorem Ipsum',
        department_id: 1,
        department: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 1,
        product_name: 'Hello',
        aisle_id: 1,
        aisle: 'Aisle Lorem Ipsum',
        department_id: 1,
        department: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    }
]

const ProductList = () => {

    const [loading, setLoading] = useState(false)
    const [products, setProducts] = useState([])
    const [error, setError] = useState('')

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true)
                // setProducts(temps)
                const backend = import.meta.env.VITE_BACKEND_BASEURL
                const request = `${backend}/detailed-products`
                const response = await axios.get(request)
                setProducts(response.data.products)
            }catch(error){
                setError(error.response)
                setProducts([])
            }finally{
                setLoading(false)
            }
        }
        fetchData()
    }, [])

    if(loading){
        return (
            <div className='w-full h-full flex justify-center items-center text-base'>
                Fetching Products from Database ...
            </div>
        )
    }

    return (
        <div className='h-full w-full'>
            <p className='text-sm text-center text-gray-600 mb-4'><i>Disclaimer: This section only shows first 1000 products for efficiency. For more detail, please refer to <b>Evaluation section</b></i></p>
            <div className='h-auto rounded-5 grid lg:grid-cols-3 md:grid-cols-2 sm:grid-cols-1 gap-2 rounded-xl'>
                {(products == null || products.length <= 0) ?
                    <p className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">No Products yet ...</p> :
                    products.map((product) => (<ProductCard product={product} key={product.product_id}/>))
                }
            </div>
        </div>
    )
}

export default ProductList